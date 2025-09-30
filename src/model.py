import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from tqdm import tqdm

warnings.filterwarnings("ignore")


class CreditModelPipeline:
    def __init__(self, train_path: str, test_path: str):
        """
        初始化训练和测试数据路径。
        """
        self.train_path = train_path
        self.test_path = test_path
        self.SK = None  # 特征选择器
        self.train_data = None
        self.test_data = None
        self.feature_names = None

    def load_train_data(self):
        """加载训练数据并做特征选择"""
        try:
            self.train_data = pd.read_csv(self.train_path)
            self.SK = SelectKBest(f_classif, k=20).fit(
                self.train_data.drop(columns=["isDefault"]), self.train_data["isDefault"]
            )
            self.feature_names = self.SK.get_feature_names_out()
        except Exception as e:
            raise RuntimeError(f"加载训练数据失败: {e}")

    def plot_corr_heatmap(self):
        """绘制特征相关性热力图"""
        try:
            corrmat = self.train_data[self.feature_names].corr().T
            corrmat = abs(corrmat)
            plt.figure(figsize=(40, 18))
            sns.heatmap(corrmat, vmax=0.8, square=True, cmap="YlGnBu", annot=True)
            plt.savefig("corr_heatmap.jpg")
        except Exception as e:
            print(f"绘制热力图失败: {e}")

    @staticmethod
    def train_xgb(**kwargs):
        """训练 XGBoost 模型"""
        return Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=kwargs.get("degree", 1), include_bias=False)),
                ("scaler", MinMaxScaler()),
                ("XGBR", XGBClassifier(
                    booster=kwargs.get("booster", "gbtree"),
                    random_state=7,
                    n_estimators=kwargs.get("n_estimators", 10),
                    max_depth=kwargs.get("max_depth", 6),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    min_child_weight=kwargs.get("min_child_weight", 1),
                    max_leaves=kwargs.get("max_leaves", 0),
                    verbosity=0,
                )),
            ]
        )

    @staticmethod
    def train_logistic_regression(**kwargs):
        """训练逻辑回归模型"""
        return Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=kwargs.get("degree", 1), include_bias=False)),
                ("scaler", MinMaxScaler()),
                ("LR", LogisticRegression(
                    penalty=kwargs.get("penalty", "l2"),
                    C=kwargs.get("C", 1.0),
                    solver=kwargs.get("solver", "saga"),
                    max_iter=int(1e6),
                )),
            ]
        )

    @staticmethod
    def train_rf(**kwargs):
        """训练随机森林模型"""
        return Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("RF", RandomForestClassifier(
                    n_estimators=kwargs.get("n_estimators", 100),
                    criterion=kwargs.get("criterion", "gini"),
                    max_depth=kwargs.get("max_depth"),
                    max_features=kwargs.get("max_features", "sqrt"),
                    min_samples_split=kwargs.get("min_samples_split", 2),
                    min_impurity_decrease=kwargs.get("min_impurity_decrease", 0),
                )),
            ]
        )

    @staticmethod
    def train_svm(Xtrain, Xtest, Ytrain, Ytest):
        """训练 SVM 模型并评估"""
        try:
            model = LinearSVC()
            model.fit(Xtrain, Ytrain)
            y_pred = model.predict(Xtest)
            score = roc_auc_score(Ytest, y_pred)
            print("SVM AUC:", score, "SVM acc:", model.score(Xtest, Ytest))
            return model, score
        except Exception as e:
            print(f"SVM 训练失败: {e}")
            return None, None

    @staticmethod
    def train_lgb(self, **kwargs):
        """
        训练 LightGBM 模型
        """
        return Pipeline([
            ("poly_features", PolynomialFeatures(degree=kwargs.get("degree", 1), include_bias=False)),
            ("scaler", MinMaxScaler()),
            ("LGB", LGBMClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", -1),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=7
            ))
        ])

    @staticmethod
    def train_knn(self, n_neighbors=5):
        """
        训练 KNN 模型
        """
        return Pipeline([
            ("scaler", MinMaxScaler()),
            ("KNN", KNeighborsClassifier(n_neighbors=n_neighbors))
        ])

    def cross_result(self, estimator, n_splits=5):
        """交叉验证结果"""
        try:
            X = self.train_data[self.feature_names]
            y = self.train_data["isDefault"]
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=7)
            result = cross_validate(
                estimator, X, y, cv=cv, scoring="roc_auc", return_train_score=True
            )
            return result
        except Exception as e:
            print(f"交叉验证失败: {e}")
            return None

    def hyperopt_search(self, max_evals=50):
        """使用 Hyperopt 搜索最优参数"""
        X, y = self.train_data[self.feature_names], self.train_data["isDefault"]

        def objective(params):
            try:
                xgb = self.train_xgb(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]),
                    learning_rate=params["learning_rate"],
                    min_child_weight=int(params["min_child_weight"]),
                )
                cv = KFold(n_splits=5, shuffle=True, random_state=7)
                result = cross_validate(xgb, X, y, cv=cv, scoring="roc_auc")
                return -np.mean(result["test_score"])
            except Exception as e:
                print(f"调参过程中出错: {e}")
                return np.inf

        space = {
            "n_estimators": hp.quniform("n_estimators", 5, 50, 2),
            "max_depth": hp.quniform("max_depth", 3, 12, 1),
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.3, 0.01),
            "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
        }

        trials = Trials()
        best_params = fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            early_stop_fn=no_progress_loss(5),
        )
        print("最优参数:", best_params)
        return best_params

    def save_model(self, model, filename):
        """保存模型到文件"""
        try:
            joblib.dump(model, filename)
            print(f"模型已保存: {filename}")
        except Exception as e:
            print(f"保存模型失败: {e}")
            
    def load_model(self, model_path, params=None):
        """
        加载模型，并可选地设置参数
        :param model_path: 模型文件路径
        :param params: dict 类型，更新模型参数
        """
        try:
            model = joblib.load(model_path)
            if params:
                model.set_params(**params)
            return model
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    def load_test_data(self):
        """加载并预处理测试数据"""
        try:
            self.test_data = pd.read_csv(self.test_path)
            self.test_data["issueDate"] = pd.to_datetime(self.test_data["issueDate"], format="%Y-%m-%d")
            self.test_data["earliesCreditLine"] = pd.to_datetime(
                self.test_data["earliesCreditLine"], errors="coerce"
            )
            startdate = datetime.datetime.strptime("2007-06-01", "%Y-%m-%d")
            self.test_data["issueDateDT"] = (self.test_data["issueDate"] - startdate).dt.days

            df = self.test_data[self.feature_names].copy()

            if "grade" in df.columns:
                le = LabelEncoder()
                df["grade"] = le.fit_transform(df["grade"])
                joblib.dump(le, "grade_label_encoder.model")

            if "subGrade" in df.columns:
                df["subGrade"] = df["subGrade"].apply(lambda x: ord(x[0]) - 64).astype(int) * 10 + df["subGrade"].str[1].astype(int)

            df = df.fillna(df.median())
            return df
        except Exception as e:
            raise RuntimeError(f"加载测试数据失败: {e}")
    def load_model(self, model_path, params=None):
        """
        加载模型，并可选地设置参数
        :param model_path: 模型文件路径
        :param params: dict 类型，更新模型参数
        """
        try:
            model = joblib.load(model_path)
            if params:
                model.set_params(**params)
            return model
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")

    def predict_single(self, model, sample):
        """
        对单个样本做预测
        :param model: 已加载的模型
        :param sample: dict / pandas.Series / numpy.array
        :return: 概率值或预测类别
        """
      
        # 保证是 dataframe 格式
        if isinstance(sample, dict):
            df = pd.DataFrame([sample])[self.feature_names]
        elif isinstance(sample, pd.Series):
            df = pd.DataFrame([sample.to_dict()])[self.feature_names]
        elif isinstance(sample, np.ndarray):
            df = pd.DataFrame([sample], columns=self.feature_names)
        else:
            raise TypeError("样本输入必须是 dict / pandas.Series / numpy.ndarray")

        try:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(df)[:, 1][0]
            elif hasattr(model, "decision_function"):
                score = model.decision_function(df)[0]
                # 归一化成 [0,1]
                return (score - score.min()) / (score.max() - score.min())
            else:
                return model.predict(df)[0]
        except Exception as e:
            raise RuntimeError(f"单样本预测失败: {e}")

    def predict_test(self, model_files):
        """对测试集进行预测 (支持XGB, RF, LR, SVM)"""
        df = self.load_test_data()
        result = pd.DataFrame()
        result["id"] = self.test_data["id"]
        
        for model_name in tqdm(model_files):
            try:
                model = joblib.load(model_name)
                if hasattr(model, "predict_proba"):
                    result[model_name + "_PRE"] = model.predict_proba(df)[:, 1]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(df)
                    result[model_name + "_PRE"] = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    raise AttributeError("模型不支持概率输出")
            except Exception as e:
                print(f"预测失败 {model_name}: {e}")
        return result


if __name__ == "__main__":
    # 使用示例
    pipeline = CreditModelPipeline(
        train_path="/data/lijunlin/project/Linear/SGD/data/clear_train_data_with_out_time.csv",
        test_path="/data/lijunlin/project/Linear/SGD/data/testA.csv",
    )

    pipeline.load_train_data()
    pipeline.plot_corr_heatmap()

    # 训练 XGB 模型并交叉验证
    xgb_model = pipeline.train_xgb(n_estimators=20, max_depth=6, learning_rate=0.1)
    print(pipeline.cross_result(xgb_model))
 
    # 保存模型
    pipeline.save_model(xgb_model, "XGB.model")

    # 调参
    pipeline.hyperopt_search(max_evals=20)

    # 预测
    predictions = pipeline.predict_test(["XGB.model"])
    print(predictions.head())
