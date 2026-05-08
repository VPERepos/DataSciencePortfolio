# Customer Behavioural Segmentation.
## Problem definition.
This project is based on the UCI Online Retail II dataset.

The goal is to separate customers into interpretable and meaningfull segments, based on their buying activities, for further bussiness decisions on actions for each customer group.

## Technologies and Models Used
In order to define how many clusters are their in data and how good one can separate them, an automatic density based segmentation algorithm **HDBSCAN** was used. Then using the same **Manhattan distance metric** a **K-Medoids** segmentation was conducted for different numbers of clusters deriving **Clustering Inertia** and **Clustering Silhouette** curves. Then the final **K-Medoids** clustering was conducted with the number of clusters derived from the three analytical methods. The data were loaded and manipulated by **Pandas** and **NumPy**  libraries.