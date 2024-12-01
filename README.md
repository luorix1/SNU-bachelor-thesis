# Foot-mounted IMU Real-time Stride Time (FIRST) Estimation

## Overview
Repository containing all code for SNU Electrical and Computer Engineering Bachelor Thesis

## Abstract
This study explores stride time prediction methods for dynamic gait analysis, focusing on baseline, indirect, and direct approaches. A custom postprocessing and graphical user interface (GUI) were designed for annotating foot IMU data, facilitating accurate data analysis. The evaluation reveals that the direct stride time prediction method, utilizing Long Short-Term Memory (LSTM) networks, achieves the highest overall accuracy, with an $R^2$ of 0.8422 and a mean absolute error (MAE) of 0.05692 at an optimal window size of 18 frames. This approach leverages sequential data processing to capture temporal dependencies and provides reliable predictions even for irregular gait patterns. The indirect method, utilizing dense gait phase predictions as inputs to a linear regression model, also performs well, offering a flexible and robust alternative. Although the direct method outperforms the indirect approach in terms of accuracy, the tradeoff between latency and performance, as seen in the direct method's reliance on swing phase start, presents challenges. To mitigate this, future work will focus on incorporating ipsilateral foot data during the contralateral foot's stance phase to reduce delays. Additionally, testing against the baseline for various gait scenarios revealed that our methods significantly outperform the baseline in non-steady-state gait scenarios while remaining comparable in steady-state cases. These results underscore the potential of machine learning techniques in advancing gait analysis and improving dynamic stride time prediction.

## Figures
Coming Soon

## Paper Link
Coming Soon
