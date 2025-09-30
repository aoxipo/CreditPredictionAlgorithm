from src.model import CreditModelPipeline  # 假设类保存为 pipeline.py

# ===============================
# 1. 初始化 & 加载数据
# ===============================
pipeline = CreditModelPipeline(
    train_path="train.csv",
    test_path="test.csv"
)

pipeline.load_train_data()  # 加载训练数据


# ===============================
# 2. 训练 + 保存模型
# ===============================

# 2.1 XGBoost
xgb_model = pipeline.train_xgb(n_estimators=50, max_depth=6, learning_rate=0.1)
print("XGB CV Result:", pipeline.cross_result(xgb_model))
pipeline.save_model(xgb_model, "XGB.model")

# 2.2 Random Forest
rf_model = pipeline.train_RandomForest_Classifier(n_estimators=100, max_depth=8)
print("RF CV Result:", pipeline.cross_result(rf_model))
pipeline.save_model(rf_model, "RF.model")

# 2.3 Logistic Regression
lr_model = pipeline.train_logistic_regression(degree=1, penalty="l2", C=1.0)
print("LR CV Result:", pipeline.cross_result(lr_model))
pipeline.save_model(lr_model, "LR.model")

# 2.4 SVM (带概率输出)
svm_model = pipeline.train_svm(kernel="linear", C=1.0)
print("SVM CV Result:", pipeline.cross_result(svm_model))
pipeline.save_model(svm_model, "SVM.model")


# ===============================
# 3. 加载 + 修改参数
# ===============================
xgb_loaded = pipeline.load_model("XGB.model", params={"XGBR__n_estimators": 200})
rf_loaded  = pipeline.load_model("RF.model", params={"RF__max_depth": 10})
lr_loaded  = pipeline.load_model("LR.model", params={"LR__C": 0.5})
svm_loaded = pipeline.load_model("SVM.model", params={"SVC__C": 2.0})

print("四个模型均已加载并可修改参数")


# ===============================
# 4. 预测整个测试集
# ===============================
result_df = pipeline.predict_test(["XGB.model", "RF.model", "LR.model", "SVM.model"])
print("测试集预测结果：")
print(result_df.head())
# 输出示例：
#    id   XGB_PRE   RF_PRE   LR_PRE   SVM_PRE
# 0  1    0.1234    0.2123   0.5432   0.6543
# 1  2    0.4321    0.3312   0.6212   0.4890


# ===============================
# 5. 预测单个样本
# ===============================
sample = {
    "loanAmnt": 15000,
    "term": 36,
    "interestRate": 12.5,
    "grade": "B",
    "subGrade": "B3",
    "employmentLength": 10,
    "issueDateDT": 3650,
    # ⚠️ 必须包含训练时选中的所有特征字段
}

print("\n单样本预测结果：")
for model_name in ["XGB.model", "RF.model", "LR.model", "SVM.model"]:
    model = pipeline.load_model(model_name)
    score = pipeline.predict_single(model, sample)
    print(f"{model_name} -> {score:.4f}")
