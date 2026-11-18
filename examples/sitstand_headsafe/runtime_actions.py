"""Explicitly configured deployment targets; importing this reads no calibration."""
from dataclasses import dataclass
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction,JointPositionActionCfg
if __package__:
    from .calibration import NAMES,SIT,load_calibration
    from .runtime_state import TargetState,VoltageState
else:
    from calibration import NAMES,SIT,load_calibration
    from runtime_state import TargetState,VoltageState

class RuntimeAction(JointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        rows=load_calibration(cfg.calibration)
        assert list(self.target_names)==NAMES
        assert torch.allclose(torch.as_tensor(self._scale),torch.ones_like(torch.as_tensor(self._scale)))
        self.home=self._offset[0].detach().clone()
        self.state=TargetState(self.num_envs,self.home.tolist(),[r['min_rad'] for r in rows],[r['max_rad'] for r in rows],self.device)
        self.voltage=VoltageState(self.num_envs,self.device)
        self.ticks=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.bam=[act for act in self._entity.actuators if hasattr(act,'vin_tensor')]
        assert len(self.bam)==1
        self.zero=torch.tensor([r['zero_raw'] for r in rows],device=self.device)
        self.direction=torch.tensor([r['direction'] for r in rows],device=self.device)

    def process_actions(self,actions):
        act=self.bam[0]
        vin=act.vin_tensor.reshape(-1).clone()
        if act.vin_drop_gain is not None:
            vin-=act.vin_drop_gain.reshape(-1)*act._prev_motor_torque.abs().sum(-1)
            if act.cfg.vin_min is not None:vin.clamp_(min=act.cfg.vin_min)
        self.voltage.update(vin,self.ticks%50==0)
        self._raw_actions.copy_(actions)
        target=self.state.step(actions,self.voltage.multiplier)
        raw=self.zero+self.direction*target*(4096/(2*torch.pi))
        raw=torch.sign(raw)*torch.floor(raw.abs()+.5)
        self._processed_actions=((raw-self.zero)*self.direction*(2*torch.pi/4096)).to(actions.dtype)
        self.ticks+=1

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'state'):
            self.state.reset(env_ids)
            self.voltage.reset(env_ids)
            self.ticks[slice(None) if env_ids is None else env_ids]=0

@dataclass(kw_only=True)
class RuntimeActionCfg(JointPositionActionCfg):
    calibration: str = ''
    def build(self,env):return RuntimeAction(self,env)
