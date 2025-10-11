# Event-Aware Graph WaveNet for Traffic Forecasting

This repository contains the PyTorch implementation for the research project "Dynamic Event-Influence Modeling for Traffic Forecasting using Attention-Modulated Graph WaveNet".

The project introduces a novel Graph Neural Network architecture designed to improve traffic forecasting, particularly for non-recurrent congestion caused by events like sports games or concerts. The core innovation is a **Dynamic Graph Attention (DGA)** module, which uses cross-attention to learn an event-driven relational bias. This allows the model to explicitly and dynamically adjust its understanding of the spatial relationships between traffic sensors in real-time based on external event data.

---

## Datasets

Data for this project available at the following Google Drive: [PEMS_Events](https://drive.google.com/drive/folders/1V3lT6yadOIGauxszsDLG2NcOslH8_6-z?usp=sharing)

## Acknowledgments and Citations

This work is built upon the foundational concepts of several key research papers and datasets. We gratefully acknowledge the authors and data providers for their significant contributions.

* **Graph WaveNet**: The core spatio-temporal architecture is based on this work.
    ```
    @inproceedings{wu2019graph,
      title={Graph wavenet for deep spatial-temporal graph modeling},
      author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
      booktitle={International joint conference on artificial intelligence (IJCAI)},
      year={2019}
    }
    ```

* **PEMS-BAY Dataset**: The traffic data used for training and evaluation was collected by the Caltrans Performance Measurement System (PeMS).

* **Sports Reference**: The sports event schedule data for 2017 was sourced from [https://www.sports-reference.com](https://www.sports-reference.com).