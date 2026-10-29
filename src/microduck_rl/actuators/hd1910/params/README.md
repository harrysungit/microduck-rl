# HD-1910 BAM parameters

`1910_m6.json` is BAM's M6 friction model ("quadratic directional") fitted for
the Feetech HD-1910, copied byte for byte from LuwuDynamics/xgoduck_rl
(Apache-2.0), `src/mjlab_microduck/robot/xgoduck/params/1910_m6.json` at
`326d77a1122870bdefa2c36403937502c958e69c`; sha256
`ef2d51adfb1cc0831b9b02ca19aa9176fceecdf725148d084378d7d9a64afceb`.

- `"actuator": "sts3215"` selects BAM's Feetech control law.
- `max_velocity` = 100 rad/s turns the firmware goal slew off; it is not the
  servo's no-load speed.
- `q_offset` and `command_delay` are fit terms, not hardware zero offsets or
  mjlab's control lag.
- Derived from the fit: about 0.91 N·m stall torque at 7.4 V (datasheet
  1.47 N·m). Feetech's own fit (ftservo/ftservo-bam) gives about 1.37 N·m but
  needs a different BAM actuator class; neither is validated on our servos.
