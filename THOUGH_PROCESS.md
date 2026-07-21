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

### 20/07/26 


I created the game a while ago , it was very hard to understand but right now I have 

- conversion rate computated accross all of the products ( more on this later)
- buybox percent - perhaps I should also only consider top performing products 


I developed 3 agents TQ: DQN and PPO - these are then trained againts static or dynamic agents 

The dynamic gave many issues so I decided to see if pretraining with a static model would yield better results - it did but still doesnt behave how I would like it to 

next step would be to train on real top performing producst before doing the reinforcement step with itself 
I need to also see the effects of the new scraped data of our commaxx products through the scraper such that I can see whether the relavance did in fact increase. 




I asked ai to change the query such thqt it only extract top performing products since a lot of them the company doesnt maintain


### 21/07/26 

I think that i can improve the prediction of the demand based on the predicted unit as probabilistic thresholds instead of raw counts - Negative Binomial distribution 


I am starting to realize that the task of prediciting the reward might be harder than what I thought , maybe instead of simulating , I could just find the closest ten products to it and weighted average those properties - f w would like to analyze the headphones then you will find the products who are most similar to it and weigh its impact on the final price / output 




I am at a road block because the demand prediction section is not working properly so I need to find a way to make the system predict demand more accurately 

https://www.dii.uchile.cl/wp-content/uploads/2014/06/Modeling+Pricing+Strategies+Using+Game+Theory+and+Support+Vector+Machines.pdf

I found this article by gods grace Imma read it and then see what happens


what I realized is that the core issue that I have is the profit model , I do not have a solid reward system that will allow for good agent v agent training 

  units = self.expected_demand(self.own_price)
        if self.stochastic:
            units = self._draw_demand(units)   # Negative Binomial (mean-preserving, overdispersed)
        reward = self._profit(self.own_price, units)



    the thing is that the reward comes from the units sold which then I guess is a poor demand prediction model even though the plots dont look unreasonable