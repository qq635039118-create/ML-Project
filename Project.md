
# Project

### Forming groups

Each group 6 students.

Forming groups:
- https://docs.qq.com/doc/DT2xqVHphanhGUWpR (Login with WeChat by scanning the QR code)


### What's expected of your video
- Length of video >= 10 min
- If possible, make it fun (Because life is good). 
- If possible, make it fancy (Because you are young). 
- And yes, your video should be presented in English. 

### Submission of your work

1. Create a folder, in which you put:
    - the video
    - the source code
    - a txt/markdown file indicating 
        - what's the task of each team member
        - the estimated workload/contribution percentage of each team member 
    - a txt/markdown file indicating 
        - the URL of your GitHub repository for hosting your code
        - prof will check the commit history of your GitHub repo to see how each team member is contributing  
1. Zip the folder
1. Upload the zip file to Google Drive
1. Send the sharing link to the prof, by PRIVATE WeChat or by Email
    - Therefore, in the WeChat/Email message, there are no attached files, just an Google Drive URL.

For each team, just one submission of the work is necessary, by one member of your team.

Deadline for submission:
- The First Friday of the 14 days of Exam Weeks of SHU, 23:59.
- For 2026, it's ?? ??/?? 23:59.

### Gallery of Final project videos

Bilibili videos:
- https://space.bilibili.com/472463946/lists/1487100?type=season
- https://www.bilibili.com/video/BV1VaokBDEnV

Be default, you agree to share your videos on Bilibili by the prof as well.



## Step 1 of the project (FOR EACH STUDENT. NO TEAM UP YET):

Use an AI tool (e.g. OpenClaw),

go to:

https://github.com/zhangqi444/open-forge

DO EVERYTHING WITH AI !

give it a star; fork this repo to your own GitHub account; 
Ideally, use WSL 2 or Docker for Win,

for macos, use Docker.

Or, you can use a cloud VM.

This project is for each of the students. So no need to team up.

Again, the idea is to DO EVERYTHING WITH AI, even for forking, even for giving a star.

# Step 2:


# Project Topics for Students

## 1. Speaker Diarization, Cross-Speech, and LLM + ASR Synergy

One important direction I would like you to explore is:

* Speaker diarization
* Multi-speaker overlapping speech (cross-speech)
* The synergy between Large Language Models (LLMs) and Automatic Speech Recognition (ASR)

Please read the `xutong_paper.pdf` file located in the `./project/` folder. Based on the existing work presented in that paper, I encourage you to:

* Explore related research topics
* Identify limitations or missing components
* Propose new ideas or improvements
* Design and test more advanced or innovative approaches

Maybe you can integrate RAG (Retrieval-augmented generation) into the pipeline/loop as well.

You are not limited to reproducing existing work. The goal is to think critically and explore what could be done next.

---

## 2. Deploying ASR Models on Mobile Devices

The second direction is focused on deploying ASR models on local mobile devices such as:

* iPhones
* Android phones

Possible exploration topics include:

* Benchmarking different ASR models
* Comparing latency, memory usage, and inference speed
* Evaluating recognition quality across multiple languages, especially:

  * English
  * French
  * Chinese
* Studying on-device inference constraints
* Investigating quantization, compression, or lightweight architectures

You may also explore research-oriented improvements, such as:

* Fine-tuning existing ASR models for mobile deployment
* Optimizing models for edge devices
* Publishing improved models on Hugging Face if you obtain strong results

Maybe you can integrate RAG (Retrieval-augmented generation) into the pipeline/loop as well.

---

## 3. Local Audio Preprocessing for Better ASR Performance

Another very interesting topic is local audio preprocessing on smartphones or PCs before the ASR stage.

Examples include:

* Noise reduction / denoising
* Signal enhancement
* Audio clustering
* Voice activity detection
* Signal processing techniques
* Echo cancellation
* Beamforming or microphone optimization

The objective is to improve downstream ASR performance from an engineering perspective.

You should investigate whether preprocessing pipelines can significantly improve:

* Recognition accuracy
* Robustness in noisy environments
* Real-time performance
* Multi-speaker recognition quality

---

## 4. Fun and Experimental Topic — “Pet Translation Device”

Another fun and creative direction is something similar to the following concept:

“Pet Translation Device”

The idea is that a small device attached to a pet collar could capture sounds made by cats or dogs and “translate” them into text, while also synchronizing generated voice/chat records on a smartphone.

Related references:

* 新浪新闻报道：
  [https://finance.sina.cn/tech/2026-05-17/detail-inhyerim1837213.d.html](https://finance.sina.cn/tech/2026-05-17/detail-inhyerim1837213.d.html)

* YouTube Short:
  [https://www.youtube.com/shorts/L88QMSnlLjM](https://www.youtube.com/shorts/L88QMSnlLjM)

This topic is intentionally exploratory and creative. You may combine:

* Audio analysis
* Classification models
* Behavioral interpretation
* LLM-based interaction systems
* Edge AI
* Mobile applications

---

# Additional Notes

Inside the `./project/` folder, you will find the following files:

* `AdamMaytoussi.pdf`
* `AdamMaytoussi.pptx`
* `AdamMaytoussi.url`

These files are **not directly related** to your project topic. However, they demonstrate the expected quality standard for project organization, presentation, and technical work.

Please note:

* Adam Maytoussi’s project group only had **two students**
* Your group contains **six students**

Therefore, your project is expected to:

* Cover more material
* Demonstrate deeper technical exploration
* Reach a significantly higher standard in both engineering and research quality

---

# Development Tools

You are strongly encouraged to use:

* AI-assisted IDEs
* AI coding assistants
* AI agents
* Modern development and experimentation workflows

These tools can help accelerate research, implementation, debugging, and experimentation.

However, you should go beyond simple programming or straightforward API usage. I would like to see deeper insights that push beyond current AI understanding — including aspects that may still be unknown or poorly understood. Your experiments should aim to uncover compelling stories, subtle behaviors, unexpected limitations, nuanced patterns, meaningful engineering trade-offs, or insightful observations about how these systems actually behave in real-world settings. The goal is not merely to build something functional, but to explore, analyze, and reveal deeper understanding through thoughtful experimentation and rigorous engineering.

you might need GPU access, you can go for https://www.autodl.com/home or other platforms.

