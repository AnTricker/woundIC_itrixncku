423: 

### side sync by git 
- req :  a project save in 3 place, must sync every time i update in each place (usually, the model training PC)
    - my laptop :  codex AI & mainly develop code/prompt here, and upload
    - model training PC : only for inference(data is here), just need sync and run
    - remote VM : for boss, required update experiment progress immediate
    
    tool A - git & github : for code & text result, can sync by github
    
    my laptop & training PC, when test nedd many unnecessary push/pull, sometimes directly develop on training PC is more cofident  
    
    some training’s env/setting…. so called”local thing”, won’t need to update
    
    tool B - google drive : for big data, will update manually (rare, not that often)
    
    worry: manually update will crash the git
    
    Problem: boss req to record progress time by time ⇒ that is, github should be 堆積; instead of 覆蓋, and this 堆積 must be tidy
    
### image-capitoning plan
- Plan A
    - local ollama VQA
    - VLM first & LLM next
    - Domain transfer :
        - refer to LLaVA-med, find the "臨床醫學" textbook(my field), to have high-quality background info
        - problem : must extract the target wound type’s info, since model’s context window might not that big
        - if want use the full domain-knowledge, that is fine-tune
    - prompt engineering
        - refer to VLM-AutoDrive ,it got many different prompt methods(image-captioning, VQA, MCQ, CoT….etc).
        - First, try with the example it give in “Appendix”
    - model :
        - less 10b : `gemma4:e4b` (8B)
        - more 10b : `gemma4:31b` 
- Plan B
    - inference with LLaVAmed directly

- (additionally/future) preprocessing 
    - cla: give few groundtruth(select from origin ds), tell model base on it to cla
    - ...more

- evaluation score

- choose validate%, generate baseline with `Gemini cloud`

---