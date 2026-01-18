# QCA: Query- and Content-Aware Keyframe Selection for Efficient Long Video Understanding
This is the official implementaion of paper **QCA: Query- and Content-Aware Keyframe Selection for Efficient Long Video Understanding**
## Abstract
Video understanding is often plagued by severe temporal redundancy, where processing dense frame sequences is both semantically inefficient and computationally expensive.
This challenge is further amplified when only a small subset of frames is truly relevant to the given query.
In this paper, we propose a \textbf{Q}uery- and \textbf{C}ontent-\textbf{A}ware (\textbf{QCA}) keyframe selection framework that can select a compact yet information-rich set of frames from long videos.
QCA first partitions the video into temporal segments and evaluates the information contribution of each segment by jointly modeling the query-frame semantic matching degree and segment content deviation.
The normalized contribution scores dynamically determine the keyframe budget allocation for each segment. Within each segment, QCA anchors on the most query-relevant frame and iteratively incorporates additional frames to maximize diversity while maintaining high semantic relevance to the query.
Crucially, our method requires no additional training and can be seamlessly integrated into existing Video-LLMs. Extensive experiments across multiple long video understanding benchmarks demonstrate that our proposed approach achieves state-of-the-art performance and has strong generalization ability. For instance, QCA achieves 67.8\% on LongVideoBench using 128 frames, while GPT-4o achieves 66.7\% using 256 frames.
## Overview
