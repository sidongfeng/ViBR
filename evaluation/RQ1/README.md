## RQ1: Performance of Action Segmentation

To answer RQ1, we first evaluate
the effectiveness of our approach in accurately
segmenting user action scenes from GUI recordings.
We evaluate segmentation performance using
three standard scene boundary detection metrics: precision,
recall, and F1-score.
For all metrics, a higher value represents better performance.
We compare our method against five state-
of-the-art approach, which are widely used for video segmentation, e.g., 
PySceneDetect, Hecate, TransNetV2, GIFdroid, and GPT-4o.

### Setup

**_PySceneDetect_**

1. Requires ffmpeg/mkvmerge for video splitting support. Windows builds (MSI installer/portable ZIP) can be found on the [download](https://www.scenedetect.com/download/) page.

2. Quick install of the tool
```
pip install scenedetect[opencv] --upgrade
```


3. Split GUI recording video on each fast cut using ffmpeg:
```
scenedetect -i <video.mp4> split-video
```

**_Hecate_**
1. Hecate has one dependency: OpenCV library with an FFMPEG support. You will need to install the library!

2. Build the Hecate tool by running the following command:
```
$ git clone https://github.com/yahoo/hecate.git
$ cd hecate
$ vim Makefile.config
 - Set INCLUDE_DIRS and LIBRARY_DIRS to where your 
   opencv library is installed. Usually under /usr/local.
 - If your OpenCV version is 2.4.x, comment out the line 
   OPENCV_VERSION := 3
 - Save and exit
$ make all
$ make distribute
```

3. Check if compile successfully
```
distribute/bin/
```

4. Detect the scene boundary of GUI recording:
```
$ ./distribute/bin/hecate -i <video.mp4> --print_shot_info  --print_keyfrm_info
```


**_TransNetV2_**
1. Install requirements
```bash
pip install tensorflow==2.1
apt-get install ffmpeg
pip install ffmpeg-python pillow
```

> Note `transnetv2-weights` directory contains files in git-lfs.
> You may need to install git-lfs and run `git lfs pull` from the root directory of the repository
> (or you can download `transnetv2-weights` directory manually).

2. Git clone TransNetV2
```
git clone https://github.com/soCzech/TransNetV2.git
python setup.py install
```

3. Infer the scene boundary of GUI recording:
```
python transnetv2.py <video.mp4> [--visualize]
```

**_GIFdroid_**
1. Install prerequisites
* Python 3.6.9 installed

2. GIFdroid installation
* Ensure that the environment you are running in is operating with Python 3.6.9.
* Current Option:
    * Clone the repository [here](https://github.com/gifdroid/gifdroid.git), navigate to gifdroid directory, and execute `pip install -r requirements.txt`. Please make sure you have installed all the requirements.

3. To obtain the UTG, we use Droidbot, please refer [here](https://github.com/honeynet/droidbot/tree/master) for Droidbot installation

4. Infer the scene boundary of GUI recording:
* Input requirements:
    * video
    * utg: GUI transition graph in json format depicting the screenshots transitions
    * artifact: screenshots in UTG
* run `python main.py --video=<filename> --utg=<utg.json> --artifact=<folder> --out=<out_filename>`.

**_GPT-4o_**
1. Install requirements
- Python 3.8+
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- `ffmpeg` installed (for video processing)

2. Prepare a GPT API key.

3. Infer the scene boundary of GUI recording:
```
from openai import OpenAI
prompt = (
        "You are a helpful assistant that detects user actions in GUI recordings. "
        "Given a sequence of ordered frames sampled from the video, return the frame numbers (relative to the original video) "
        "where user interactions cause significant GUI changes. Format: a JSON list of boundary frame indices."
    )
client = OpenAI(api_key=api_key)
content = [{"type": "text", "text": prompt}]
frames = extract_frames(video, frames_dir, fps, max_frames)
for fp in frames:
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_b64(fp)}", "detail": "low"},
        }
    )

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": content}],
)
content_text = completion.choices[0].message.content
```

### Run Experiment
> Download the dataset from [https://drive.google.com/file/d/1kgtK8rAbQvaWrK_5HcLWAWq_QFRBc203/view?usp=sharing](https://drive.google.com/file/d/1kgtK8rAbQvaWrK_5HcLWAWq_QFRBc203/view?usp=sharing)

Use `run_rq1.py` to launch any single baseline on one GUI recording. Outputs are stored under `evaluation/RQ1/runs/<method>/<case>/`.

Examples:

- PySceneDetect
```
python evaluation/RQ1/run_rq1.py --video dataset/FirefoxLite-4881/video-#4881.mp4 --method pyscenedetect
```

- Hecate
```
python evaluation/RQ1/run_rq1.py --video dataset/FirefoxLite-4881/video-#4881.mp4 --method hecate --hecate-bin /path/to/distribute/bin/hecate
```

- TransNetV2
```
python evaluation/RQ1/run_rq1.py --video dataset/FirefoxLite-4881/video-#4881.mp4 --method transnetv2 --transnetv2-script /path/to/TransNetV2/transnetv2.py
```

- GIFdroid
```
python evaluation/RQ1/run_rq1.py \
  --video dataset/FirefoxLite-4881/video-#4881.mp4 \
  --method gifdroid \
  --gifdroid-main /path/to/gifdroid/main.py \
  --gifdroid-utg /path/to/utg.json \
  --gifdroid-artifact /path/to/artifact_dir \
  --gifdroid-out-name gifdroid_out.json
```

- GPT‑4o
```
export OPENAI_API_KEY=sk-...
python evaluation/RQ1/run_rq1.py \
  --video dataset/FirefoxLite-4881/video-#4881.mp4 \
  --method gpt4o \
```


### Results

<p align="center">
<img src="../../figures/rq1.png" width="65%">
</p>
<p align="center">Table: Performance comparison for action boundary segmentation.</p>


> Due to the github storage limit, we organize our dataset in JSON format in [here](./dataset.json), including the video file paths and the ground-truth annotations of the action scene boundaries.


Our approach significantly
outperforms all baselines, 7.4% in precision, 1.2% in recall, and 4.9% in F1-score over the best-performing baseline (GIFdroid). GPT-4o exhibits the
lowest performance. 
While GPT-4o demonstrates strong
semantic understanding in many tasks, it struggles with frame-
level segmentation, primarily due to its reliance on high-level
abstraction that lacks the granularity necessary for accurately
identifying scene boundaries—particularly when user actions
involve minor but functionally significant interface changes.
Similarly, the relatively low performance of TransNetV2 is also likely due to the domain discrepancy between the natural scenes it was trained and the artificial nature of GUI recordings. 

Traditional image-processing methods, such as PySceneDetect and Hecate, achieve relatively low performance.
This is because these methods rely on low-level visual heuristics, e.g., histogram differences or aesthetic scoring, that are not well-suited to the subtle and fine-grained transitions present in GUI recordings. 
Among the baselines, GIFdroid performs best. 
However, since it relies on SSIM as a perception metric, it remains limited when segmenting actions that involve large semantic differences but exhibit only minor pixel-level intensity differences.

