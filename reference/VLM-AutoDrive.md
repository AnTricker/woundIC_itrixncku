>scene: SFT& RL a VLM for autoDrive

### Understanding VQA
You are an AI visual assistant watching a single video. A detailed description summarizing the visual content of the video is provided to you. Assume that you are watching the video directly — not reading a textual description — and generate questions based on what you "see". Your task is to create multiple-choice question triplets (Question, Options, Answer) that reflect a deep understanding of the video content.

Objectives:
- Encourage comprehensive reasoning about the scene.
- Cover diverse aspects (e.g., actions, context, causality, attribution, environment).
- Reflect visual details and metadata faithfully — do not invent content not present in the description.

Focus Areas (Dimensions to Cover):
Your questions must cover at least 5–7 different dimensions, such as:
- Weather, visibility, light conditions
- Road and environmental conditions
- Motion behavior of ego and third-party vehicles
- Impact sides of involved vehicles
- Blame assessment (driver, road, third party)
- Pedestrian or cyclist actions
- Vehicle roles and types
- Scene context (e.g., traffic density, zone, location)

Mandatory Dimension:
- You must **always include one question** that asks whether the event depicted is a **collision**, **near collision**, or **normal driving**.
- This question should also attempt to infer **why** the event was categorized as such, based on visual details in the description (e.g., sudden braking, vehicle
contact, evasive maneuvers, etc.).

Guidelines for generating Question-Answer Triplets:
- Generate 5 to 7 high-quality question-option-answer triplets.
- Each question should cover a distinct dimension of the scene.
- Each question should have 3 to 4 options (labeled A, B, C, D).
- Only one correct answer per question. Provide it as a single letter (e.g., "A").
- Do not copy the exact wording from the description. Paraphrase naturally as if interpreting visual information.

Output Format:
Return a Python dictionary string in the following JSON structure:
```json
[{
"Dimension": "<dimension-1>",
"Question": "<question-1>",
"Options": "A.XXX\nB.XXX\nC.XXX\nD.XXX",
"Answer": "<A/B/C/D>"
},
]
{structured_caption}
```

---
### Reasoning VQA
You are a highly intelligent reasoning model. Your task is to watch a video from the perspective of a vehicle's dashcam. Base d on the following captions for a video,generate a challenging multiple-choice question that requires **multiple reasoning steps** and deep understanding to answer.

The question should involve as many logical steps as possible, ensuring that the answer cannot be deduced without careful ana lysis of the captions.
Provide the question with four options (A, B, C, D), clearly indicating the correct answer, and include detailed reasoning. E nsure you answer the question as if you
are directly watching the video and include detailed reasoning:
The question should be related to Goal and Intention Reasoning and should focus on the driving event.

Note: Respond as if you're directly observing the video. Infer details like responsibility, environment, and impact areas natural ly from the scene. Do not use
phrases like 'blame assessment' or refer to any metadata explicitly.

Here is everything you observe from the video:
{structured_caption}
{vlm_caption}

Output format:
QUESTION: <Your question>
OPTIONS:
A. <Option A>
B. <Option B>
C. <Option C>
D. <Option D>
ANSWER: <Correct answer (e.g., A, B, C, or D)>

REASONS:
#####
- <Reason 1>
- <Reason 2>
- <Reason 3>
- <Reason 4>
- <Reason 5>
##### (Add as many steps as needed.)

---
### Thinking Trace
You are a highly intelligent reasoning model. Your task is to watch a video from the perspective of a vehicle's dashcam and d etermine what type of driving event
occurs.

This is the video you see:
{structured_caption}
{vlm_caption}

Note: Respond as if you're directly observing the video. Infer details like responsibility, environment, and impact areas natural ly from the scene. Do not use
phrases like 'blame assessment' or refer to any metadata explicitly. Do not mention that you are reading the description but rather that you are directly observing
the video.

Based on your observation of the video, answer the following question as if you are directly watching the video:

In this video, the ego vehicle's dashcam captures a driving event. What type of driving event occurs?
A. The vehicle collides with another vehicle or object.
B. The vehicle drives normally with no incident.
C. The vehicle nearly collides but avoids an accident.

ANSWER: <Output the correct answer (A, B, or C)>