import asyncio

import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ModelGenerator:
    def __init__(self, database, train_file, test_file, target_feature, model_num):
        """Ограничения на размер датасэта и закомментированы классификаторы
        занимающие на локальной машине слишком много времени"""
        self._database = database
        self._train_df = pd.read_csv(train_file)
        if len(self._train_df.index) > 5_000:
            self._train_df = self._train_df.sample(n=5_000)
        self._test_df = pd.read_csv(test_file)
        if len(self._test_df.index) > 500:
            self._test_df = self._test_df.sample(n=500)
        self._target_feature = target_feature
        self._model_num = model_num
        self._classifiers = [
            # KNeighborsClassifier(3),
            # SVC(kernel="linear", C=0.025),
            # SVC(gamma=2, C=1),
            # GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]

    async def _fit_best_model(self):
        x = self._train_df.drop([self._target_feature], axis=1)
        y = self._train_df[[self._target_feature]]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.1
        )
        best_model = None
        f1 = 0
        for model in self._classifiers:
            print(type(model).__name__)
            await asyncio.sleep(0.5)
            model.fit(x_train, y_train.values.ravel())
            y_pred = model.predict(x_test)
            if f1_score(y_test, y_pred) > f1:
                f1 = f1_score(y_test, y_pred)
                best_model = model
        return best_model, f1

    def _preproces_features(self):
        features_to_drop = []
        for features_name in self._train_df:
            if self._train_df[features_name].dtype == 'object':
                le = preprocessing.LabelEncoder()
                self._train_df[features_name] = le.fit_transform(self._train_df[features_name].astype(str))
                try:
                    self._test_df[features_name] = le.transform(self._test_df[features_name].astype(str))
                except ValueError:
                    features_to_drop.append(features_name)
        for features_name in features_to_drop:
            self._train_df = self._train_df.drop(columns=[features_name])
            self._test_df = self._test_df.drop(columns=[features_name])

    def _find_corr_features(self):
        """Оставляем фичи с минимальной корреляцией"""
        corr = self._train_df.corr()
        df_new_1 = self._train_df[corr[self._target_feature][corr[self._target_feature] >= 0.05].index]
        df_new_2 = self._train_df[corr[self._target_feature][corr[self._target_feature] <= -0.05].index]
        df_new = pd.concat([df_new_1, df_new_2], axis=1)
        for feature_name in self._test_df:
            if feature_name not in df_new:
                self._test_df = self._test_df.drop(columns=[feature_name])
        self._train_df = df_new.fillna(df_new.median()).reindex(sorted(df_new.columns), axis=1)
        self._test_df = self._test_df.fillna(self._test_df.median()).reindex(sorted(self._test_df.columns), axis=1)

    def _write_to_csv(self, f1: float):
        path = f'/app/model_results/{self._model_num}_{f1}.csv'
        self._test_df.to_csv(path)
        self._database.set(str(self._model_num), path)

    async def predict(self):
        if self._target_feature not in self._train_df:
            self._database.set(str(self._model_num), f'Invalid target feature {self._target_feature}')
        self._preproces_features()
        self._find_corr_features()
        model, f1 = await self._fit_best_model()
        self._test_df[self._target_feature] = model.predict(self._test_df)
        self._write_to_csv(f1)
