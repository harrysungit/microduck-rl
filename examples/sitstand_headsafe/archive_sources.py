"""Resolve literal task names and inherited command builders from archived Python AST.

No source is imported or executed. Dynamic registration that cannot be proved
from the archive fails closed instead of inventing a base-task compatibility alias.
"""
import ast
from pathlib import Path

class ArchivedSources:
    def __init__(self,directory):self.directory=Path(directory);self.cache={}
    def tree(self,path):
        path=Path(path)
        if path not in self.cache:self.cache[path]=ast.parse(path.read_text(encoding='utf-8'))
        return self.cache[path]
    def module_path(self,module):
        path=self.directory/(module.rsplit('.',1)[-1]+'.py')
        if not path.is_file():raise ValueError('Unarchived source module: '+module)
        return path
    def imports(self,path):
        result={}
        for node in ast.walk(self.tree(path)):
            if isinstance(node,ast.Import):
                for alias in node.names:
                    key=alias.asname or alias.name.split('.')[0]
                    value=(alias.name,None)
                    if key in result and result[key]!=value:raise ValueError('Ambiguous archived import: '+key)
                    result[key]=value
            elif isinstance(node,ast.ImportFrom):
                for alias in node.names:
                    if alias.name=='*':continue
                    key=alias.asname or alias.name
                    value=(node.module,alias.name) if node.module else (alias.name,None)
                    if key in result and result[key]!=value:raise ValueError('Ambiguous archived import: '+key)
                    result[key]=value
        return result
    def symbol(self,path,node):
        imports=self.imports(path)
        if isinstance(node,ast.Name):
            if any(isinstance(item,(ast.ClassDef,ast.FunctionDef)) and item.name==node.id for item in self.tree(path).body):
                return Path(path),node.id
            if node.id in imports:
                module,name=imports[node.id]
                if name is not None:return self.module_path(module),name
        elif isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name):
            if node.value.id in imports:
                module,name=imports[node.value.id]
                if name is None:return self.module_path(module),node.attr
        raise ValueError('Dynamic or unarchived source reference cannot be resolved')
    def constant(self,path,name,seen=()):
        marker=(str(path),name)
        if marker in seen:raise ValueError('Cyclic archived constant')
        seen=(*seen,marker)
        assignments=[node for node in self.tree(path).body if isinstance(node,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)]
        if len(assignments)>1:raise ValueError('Reassigned archived task constants are ambiguous: '+name)
        if assignments:return self.expression(path,assignments[0].value,seen)
        imports=self.imports(path)
        if name in imports and imports[name][1] is not None:
            module,symbol=imports[name]
            return self.constant(self.module_path(module),symbol,seen)
        raise ValueError('Missing literal archived constant: '+name)
    def expression(self,path,node,seen):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):return node.value
        if isinstance(node,ast.Name):return self.constant(path,node.id,seen)
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return self.expression(path,node.left,seen)+self.expression(path,node.right,seen)
        if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name):
            imported=self.imports(path).get(node.value.id)
            if imported and imported[1] is None:return self.constant(self.module_path(imported[0]),node.attr,seen)
        raise ValueError('Task must be reconstructible from literal archived source')
    def definition(self,path,name,kind):
        found=[node for node in self.tree(path).body if isinstance(node,kind) and node.name==name]
        if len(found)!=1:raise ValueError('Missing/ambiguous archived definition: '+name)
        return found[0]
    def build_method(self,path,name,seen=()):
        marker=(str(path),name)
        if marker in seen:raise ValueError('Cyclic archived class inheritance')
        cls=self.definition(path,name,ast.ClassDef)
        methods=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='build']
        if methods:
            if len(methods)!=1:raise ValueError('Ambiguous builder')
            calls=[node.value for node in ast.walk(methods[0]) if isinstance(node,ast.Return) and isinstance(node.value,ast.Call)]
            if len(calls)!=1:raise ValueError('Dynamic command builder cannot be established')
            owner,command=self.symbol(path,calls[0].func)
            self.definition(owner,command,ast.ClassDef)
            return owner,command,[(Path(path),name),(owner,command)]
        inherited=[]
        for parent in cls.bases:
            owner,symbol=self.symbol(path,parent)
            inherited.append(self.build_method(owner,symbol,(*seen,marker)))
        if len(inherited)!=1:raise ValueError('Command builder inheritance is ambiguous')
        owner,command,trace=inherited[0]
        return owner,command,[(Path(path),name),*trace]
    def command_builder(self,path,seen=()):
        if str(path) in seen:raise ValueError('Cyclic register delegation')
        register=self.definition(path,'register',ast.FunctionDef)
        configs=[]
        for node in ast.walk(register):
            if not isinstance(node,ast.Assign) or not isinstance(node.value,ast.Call):continue
            for target in node.targets:
                if (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Attribute)
                    and target.value.attr=='commands' and isinstance(target.slice,ast.Constant) and target.slice.value=='twist'):
                    configs.append(self.symbol(path,node.value.func))
        if configs:
            if len(set(configs))!=1:raise ValueError('Multiple twist builders in registration')
            owner,name=configs[-1]
            return self.build_method(owner,name)
        delegates=set()
        for node in ast.walk(register):
            if not isinstance(node,ast.Call):continue
            try:owner,name=self.symbol(path,node.func)
            except ValueError:continue
            if name=='register' and owner!=path:delegates.add(owner)
        if len(delegates)!=1:raise ValueError('Registration has no uniquely proven command builder')
        return self.command_builder(next(iter(delegates)),(*seen,str(path)))
    def prove_registration(self,path,requested_task):
        register=self.definition(path,'register',ast.FunctionDef)
        declared=[]
        for node in ast.walk(register):
            if not isinstance(node,ast.Call) or not node.args:continue
            if isinstance(node.func,ast.Name):
                imported=self.imports(path).get(node.func.id)
                is_registration=node.func.id=='register_mjlab_task' or imported==('mjlab.tasks.registry','register_mjlab_task')
            else:is_registration=isinstance(node.func,ast.Attribute) and node.func.attr=='register_mjlab_task'
            if is_registration:declared.append(self.expression(path,node.args[0],()))
        if requested_task not in declared:
            raise ValueError('Archived registration does not prove the requested task name; delegated/dynamic task overrides require explicit review')


def actual_task(path):
    """Literal task declaration only, not an observation of a historical runtime instance."""
    return ArchivedSources(Path(path).parent).constant(Path(path),'TASK')


def training_adapter(directory,requested_task):
    archive=ArchivedSources(directory);matches=[]
    for path in sorted(Path(directory).glob('headsafe_task*.py')):
        try:task=archive.constant(path,'TASK')
        except ValueError:continue
        if task==requested_task:matches.append(path)
    if len(matches)!=1:raise ValueError('Requested task has no unique literal archived registration module')
    selected=matches[0]
    archive.prove_registration(selected,requested_task)
    owner,name,trace=archive.command_builder(selected)
    return {'evidence_kind':'static_archived_source_ast_only','runtime_instance_verified':False,
            'task_source':selected.name,'command_callable':owner.stem+'.'+name,
            'builder_resolution':[{'source':path.name,'symbol':symbol} for path,symbol in trace]}
