#!/bin/sh

# python3 evaluation.py load_books all_features run_file # count_features gender_feature proximity_feature
cd ..
echo -e "\t ===== Death Prediction ====="
echo -e "\t ===== Logistic Regression ====="
./data/trec_eval -m set_F data/deaths.qrels output/predictions_logit.run -q

echo -e "\t ===== SVC ====="
./data/trec_eval -m set_F data/deaths.qrels output/predictions_svc.run -q

echo -e "\t ===== Decision Tree ====="
./data/trec_eval -m set_F data/deaths.qrels output/predictions_dt.run -q

echo -e "\t ===== KNN ====="
./data/trec_eval -m set_F data/deaths.qrels output/predictions_knn.run -q

echo -e "\t ===== Naive Bayes ====="
./data/trec_eval -m set_F data/deaths.qrels output/predictions_naive.run -q

echo -e "\t ===== Gender Prediction ====="
./data/trec_eval -m set_F data/gender.qrels output/gender.run -q

cd src
