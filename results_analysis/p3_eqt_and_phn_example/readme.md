# User Guide
## 1.The input of phnet_exhibition come from <strong>line 537 of PhaseNet/run.npy, right after 'picks_batch'</strong>

line 537:np.save('trace_prob.npy', pred_batch)

ps:Every time we get the 'trace_prob', we should denote the line 537