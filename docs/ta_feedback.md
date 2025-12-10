Excellent progress, I'm glad to see you've trained version 1s of all your models already!
My main concern is with your baselines; specifically, I want them to be stronger. You're using an overly-simple decoder for them vs. your primary model, you haven't swept number of layers / different node initializations, etc. Since your GCN is outperforming GT, it either means that your task is not hard enough (which clearly isn't the case given your low overall %s) or that things are under/suboptimally-trained. Having these stronger baselines will also help you with your next step of which elements to incorporate into your GDIN.
Also, I'm not sure why your val and test are so different. I assume you random-split and so your val and test should be fairly comparable? Unless test is something held-out and thus from a different distribution than train+val (if so, your result is very cool). Please clarify this in your final report.
Regardless, great work, and looking forward to your final results and submission!

TA Feedback 

To-do for project:
- Improve baselines (GCN, Graph Transformer / GAT (same thing right?), Graph SAGE)
- Improve our model (GDIN) we need to establish wtf this is, because I dont know what it is in the codebase (I think technically this is our improved GCN, so lets try to make this even better and integrate other things outside GCN so we can call it our own novel architecture)
- keep track of all experiments and ablations
- make cool figures
- assemble blog post
