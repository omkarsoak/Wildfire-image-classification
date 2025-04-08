# A Novel Transfer Learning based CNN Model for Wildfire Susceptibility Prediction

[Omkar Oak](https://github.com/omkarsoak), [Rukmini Nazre](https://github.com/rukmini-17), [Soham Naigaonkar](https://github.com/s0hamn), Suraj Sawant, Amit Joshi <br>
Department of Computer Science and Engineering, COEP Technological University, Pune, India. <br>

## Abstract

Wildfires are one of the most commonly occurring natural disasters in the world, posing significant threats to ecosystems and human settlements alike. One of the most important risk mitigation strategies is to implement early warning systems by identifying the regions more susceptible to wildfires. The development of remote sensing technologies combined with the increasing success of deep learning algorithms has greatly accelerated the development of such systems. Significant research has been done so far for wildfire detection in ground level imagery using neural network classifiers, but there is a lack of research focusing on satellite imagery. 

This paper proposes a method of wildfire risk assessment of large land regions focusing on remote sensing satellite imagery. The dataset used consists of 31280 satellite images each of size 350 x 350 pixels in jpg format from the Quebec region and was built using wildfire data from Canada's Open Government Portal website to identify regions where wildfires have occurred, and satellite images of those regions before or during occurrence have been used as the wildfire class in the binary classification. 

We implemented different CNN based classification models, namely VGG16, ResNet50, Xception and InceptionV3 as well as four VGG-16 based transfer learning models. All the models were simulated on 5640 test images and their performance was compared. Our proposed transfer learning model having a three layered pyramid structure with Batch Normalization and Dropouts yielded the best results, with an accuracy of 0.9650, 0.9715 precision and an F-1 score of 0.9648 outperforming all of the traditional as well as the basic transfer learning models by a notable margin.

## Citation

If you use this code or the findings in your research, please cite our paper:

```
@INPROCEEDINGS{10593496,
  author={Oak, Omkar and Nazre, Rukmini and Naigaonkar, Soham and Sawant, Suraj and Joshi, Amit},
  booktitle={2024 5th International Conference for Emerging Technology (INCET)}, 
  title={A Novel Transfer Learning based CNN Model for Wildfire Susceptibility Prediction}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Wildfires;Biological system modeling;Transfer learning;Focusing;Predictive models;Satellite images;Remote sensing;Deep Learning;CNN;Computer vision;Transfer Learning;VGG16;Wildfire},
  doi={10.1109/INCET61516.2024.10593496}}
```

## Contact

For any questions or inquiries about this research, please open an issue on this repository or contact the corresponding author.
