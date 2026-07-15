# TESTING 

Collection of my spontaneous thoughts for the report such that I can have a sexy report for Monsieur Aniruhd. 


### 13/07/26
---


I will go with the louvain SVD technique paired with KNN 

Now what i need to do is to test I have tried to check for the modularity_K now the result I get a good in the sense that for a large K the metric shows high results as seen by the polts but it seems that the lower the K the better so I will pair this metric with another which measures the quality by the size of the clusters - bigger means better such that I can find a ratio between then and establish an optimum. 


If this process is quick enough I could in theory apply it to all of the segementation levels such that I have an automatic bets k and resolution 

I mapped out conductance to modularity and I realized that I can make a singular metric that would favourize high performance but high stability . using this I can make accurate groupings for each segementation level automatically 


What I now have to do is compute the impact of each feature on the price of the final product then weight the proximity of the product in general 


*** WHEN writing the report I need to make sure that I understand ridge and lasso and why I used the Elastic Net or something else ***



### 14/07/26
---


The SHAP model work decently, now AI suggested that instead of training on the price directly i train on the log of the price , since the data has many wide ranges, I would need to do this to reduce the space and then exponentiate it 


    price_preds = regressor.predict_price_parsed_columns(
        feature_types=f"data_files/feature_types_{search_term}.json",
        price_params_path=f"data_files/all_feature_data_{search_term}.params.json",
        log_price=True)


it is now a parameters or the predict price parsed columns 


we can do several run , we can fine tune and I am sure that we could get decently close , but then again i think we are limited by the fact that I am only using product features to predict price and no actual market analysis

- I will be taking a game theoretic approach. 


