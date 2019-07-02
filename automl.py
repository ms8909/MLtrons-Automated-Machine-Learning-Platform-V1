from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
import numpy as np
import h2o
import boto3
import os
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from random import randint
from h2o.automl import H2OAutoML
import string
import pyspark
from pyspark.sql import SparkSession
from django.conf import settings
from datetime import datetime
import random

class train_test():

    def __init__(self, ID, key):

        self.selected_columns = None
        self.y_variable = None
        # h2o.init(max_mem_size="6g")

        # initialize context
        self.param = None
        self.automlEstimator = [None, None, None, None, None]
        self.models = [["GLM", "XGBoost", "GBM", "DeepLearning", "StackedEnsemble"],
                       ["DRF", "XGBoost", "GBM", "DeepLearning", "StackedEnsemble"],
                       ["DRF", "GLM", "GBM", "DeepLearning", "StackedEnsemble"],
                       ["DRF", "GLM", "XGBoost", "DeepLearning", "StackedEnsemble"],
                       ["DRF", "GLM", "XGBoost", "GBM", "StackedEnsemble"]]

        self.project_name = 'mltrons'
        self.trained_models = []
        self.seed = randint(1, 99)
        self.c = 1

        #self.automl = None
        self.model_num = 0
        self.h2otrain = None
        self.h2otest = None

    # methods
    def set_parameter(self, key, value):
        self.param[key] = value

    def get_parameter(self, key=None):
        if key == None:
            return self.param
        return self.param[key]

    def init_models(self, projectname):
        for i in range(5):
            self.automlEstimator[i] = H2OAutoML(max_models=1, seed=self.seed, project_name=projectname + str(i),
                                                exclude_algos=self.models[i])

    # methods
    def run_automl(self, df=None, prediction_col=None, max_models=5, project_name='mltrons', problem_type=None):
        print("In automl1")
        if df == None or prediction_col == None:
            print("df or perdiction_co is set to None.")
            return False

        # update variables
        if self.automlEstimator[0] == None:
            # convert spark frame into h2o frame
            print("First time training")
            df2 = df.toPandas()
            try:
                self.sc = pyspark.SparkContext.getOrCreate()
                self.sc.stop()
            except:
                print("unable to stop spark context")

            h2o.init(max_mem_size="6g")

            hf = h2o.H2OFrame(df2)

            # split the frame and run and make variables
            splits = hf.split_frame(ratios=[0.95], seed=self.seed)
            self.h2otrain = splits[0]
            self.h2otest = splits[1]

            ## conver the column here based on the type
            if problem_type == 'Classification':
                # change the type of y column in both test and train dataset
                self.h2otrain[prediction_col] = self.h2otrain[prediction_col].asfactor()
                self.h2otest[prediction_col] = self.h2otest[prediction_col].asfactor()

            self.y_variable = prediction_col


        print("In automl2",self.automlEstimator[self.model_num])
        if self.automlEstimator[0] == None:
            # need to initialize the automl
            self.init_models(project_name)
        else:
            print("We trained before.",self.model_num )
            if problem_type == 'Classification':
                self.model_num = (self.model_num + 1) % (len(self.automlEstimator)-1)
            else:
                self.model_num = (self.model_num + 1) % len(self.automlEstimator)
            print("We are about to train",self.model_num )
            if self.model_num == 0:
                self.init_models(project_name)

        #         self.automlEstimator= H2OAutoML(max_models=max_models, seed = self.seed, project_name = project_name)
        print("In automl3",self.automlEstimator[self.model_num])
        self.automlEstimator[self.model_num].train(y=self.y_variable, training_frame=self.h2otrain,
                                                   validation_frame=self.h2otrain, leaderboard_frame=self.h2otrain)

        return True

    def get_details(self):
        details = []
        mod = []
        # models = np.array(self.automlEstimator.leaderboard.as_data_frame()['model_id'])
        models = np.array(self.automlEstimator[self.model_num].leaderboard.as_data_frame()['model_id'])
        for m_id in models:
            m = h2o.get_model(m_id)
            if m_id in self.trained_models:
                pass
            else:
                self.c += 1
                if 'DRF' in m_id:
                    x = self.get_details_DRF(m)
                    x['name'] = 'Distributed Random Forest' + '_' + str(self.c)
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)
                if 'GBM' in m_id:
                    x = self.get_details_GBM(m)
                    x['name'] = 'Gradient Boosting Machine' + '_' + str(self.c)
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)
                if 'GLM' in m_id:
                    x = self.get_details_GLM(m)
                    x['name'] = 'Generalized Linear Modeling' + '_' + str(self.c)
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)
                if 'XGBoost' in m_id:
                    x = self.get_details_XGB(m)
                    x['name'] = 'XGBoost' + '_' + str(self.c)
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)

                if 'Stacked' in m_id:
                    x = self.get_details_ST(m)
                    x['name'] = 'Stacked Ensemble' + '_' + str(self.c)
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)
                if 'Deep' in m_id:
                    x = self.get_details_DL(m)
                    x['name'] = 'Deep Learning' + '_' + str(self.c)
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)

        for i in mod:
            self.trained_models.append(i)
        return details

    def get_details_classification(self):
        details = []
        mod = []
        models = np.array(self.automlEstimator[self.model_num].leaderboard.as_data_frame()['model_id'])
        # models = np.array(self.automlEstimator.leaderboard.as_data_frame()['model_id'])
        for m_id in models:
            m = h2o.get_model(m_id)
            if m_id in self.trained_models:
                pass
            else:
                self.c += 1
                if 'GLM' in m_id:
                    print("model id is: " + str(m_id))
                    x = self.get_details_GLM_classification(m)
                    print("after glm")
                    x['name'] = m_id
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)
                elif 'Stacked' in m_id:
                    print("model id is: " + str(m_id))
                    x = self.get_details_ST_classification(m)
                    print("after stacked")
                    x['name'] = m_id
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)
                elif 'DeepLearning' in m_id:
                    x = self.get_details_dl_classification(m)
                    print("after stacked")
                    x['name'] = m_id
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)

                else:
                    print("model id is: " + str(m_id))
                    x = self.get_details_other_classification(m)
                    print("after detailed")
                    x['name'] = m_id
                    x['model'] = m
                    details.append(x)
                    mod.append(m_id)


        print ("reached here now")
        for i in mod:
            self.trained_models.append(i)
        return details

    # details for classification

    def get_details_dl_classification(self, model):
        metric = {}
        deviance_iter = {}
        lift_gain = {}
        confusion_matrix = {}
        try:
            metric = self.calculate_metrics_classification(model)
        except:
            metric = None
        print("after here 1")
        deviance_iter = self.calculate_deviance_iteration_classification(model)
        lift_gain = self.calculate_lift_gain(model)
        confusion_matrix = self.calculate_confusion_matrix(model)

        model_details = {}
        graph = {}
        model_details['Metric'] = metric
        graph['Deviance vs Iter'] = deviance_iter
        graph['lift_gain'] = lift_gain
        graph['confusion_matrix'] = confusion_matrix
        model_details['graph'] = graph

        return model_details

    def get_details_GLM_classification(self, model):
        metric = {}
        deviance_iter = {}
        lift_gain = {}
        confusion_matrix = {}
        try:
            metric = self.calculate_metrics_classification(model)
        except:
            metric = None
        print("after here 1")
        deviance_iter = self.calculate_deviance_iteration_classification(model)
        lift_gain = self.calculate_lift_gain(model)
        confusion_matrix = self.calculate_confusion_matrix(model)

        model_details = {}
        graph = {}
        model_details['Metric'] = metric
        graph['Deviance vs Iter'] = deviance_iter
        graph['lift_gain'] = lift_gain
        graph['confusion_matrix'] = confusion_matrix
        model_details['graph'] = graph

        return model_details


    def get_details_other_classification(self, model):
        metric = {}
        loss_trees = {}
        lift_gain = {}
        confusion_matrix = {}
        variable_importance = {}

        try:
            metric = self.calculate_metrics_classification(model)
        except:
            metric = None
        loss_trees = self.calculate_vali_loss_classification(model)
        lift_gain = self.calculate_lift_gain(model)
        confusion_matrix = self.calculate_confusion_matrix(model)
        variable_importance = self.calculate_variable_importance_class(model)

        model_details = {}
        graph = {}
        model_details['Metric'] = metric
        graph['Loss vs trees'] = loss_trees
        graph['lift_gain'] = lift_gain
        graph['confusion_matrix'] = confusion_matrix
        graph['Variable Importance'] = variable_importance
        model_details['graph'] = graph

        return model_details

    def get_details_ST_classification(self, model):
        metric = {}
        lift_gain = {}
        confusion_matrix = {}

        try:
            metric = self.calculate_metrics_classification(model)
        except:
            metric = None
        lift_gain = self.calculate_lift_gain(model)
        confusion_matrix = self.calculate_confusion_matrix(model)

        model_details = {}
        graph = {}
        model_details['Metric'] = metric
        graph['lift_gain'] = lift_gain
        graph['confusion_matrix'] = confusion_matrix
        model_details['graph'] = graph

        return model_details

    def calculate_confusion_matrix(self, model):
        c = model.confusion_matrix()
        d = c.table.as_data_frame()
        dic = d.to_dict()
        return dic

    def calculate_lift_gain(self, model):
        # result = {}
        # lg = model.gains_lift().as_data_frame()
        # result['group'] = list(lg['group'])
        # result['cumulative_lift'] = list(lg['cumulative_lift'])
        # result['cumulative_gain'] = list(lg['cumulative_gain'])
        # return result

        result = {}
        lg = model.gains_lift().as_data_frame()
        result['group'] = list(lg['group'])
        result['cumulative_lift'] = list(lg['cumulative_lift'])
        result['cumulative_data_fraction'] = list(lg['cumulative_data_fraction'])
        result['cumulative_capture_rate'] = list(lg['cumulative_capture_rate'])

        x = model.model_performance()
        result['False Positive Rate'] = x.fprs
        result['True Positive Rate'] = x.tprs
        return result

    def calculate_vali_loss_classification(self, model):
        result = {}
        result['Training Log Loss'] = list(model.scoring_history()['training_logloss'])
        result['Validation Log Loss'] = list(model.scoring_history()['validation_logloss'])
        result['Number of Trees'] = list(model.scoring_history()['number_of_trees'])
        return result

    def calculate_deviance_iteration_classification(self, model):
        result = {}
        result['Deviance Train'] = list(model.scoring_history()['deviance_train'])
        result['Deviance Test'] = list(model.scoring_history()['deviance_test'])
        result['Iteration'] = list(model.scoring_history()['iteration'])
        return result

    def calculate_variable_importance_class(self, model):
        result = {}
        variable_names = []
        importance = []
        for every in model.varimp():
            variable_names.append(every[0])
            importance.append(every[1])

        result["variables"] = variable_names
        result["importance"] = importance
        return result

    def calculate_metrics_classification(self, model):
        cv = model.cross_validation_metrics_summary().as_data_frame()
        result = {}
        metric = []
        value = []
        cv_array = np.array(cv)
        for i in cv_array:
            metric.append(i[0])
            value.append(float(i[1]))
        result['name'] = metric
        result['value'] = value
        return result

    # model details for regression and time series

    def get_details_DRF(self, model):
        return self.get_details_GBM(model)

    def get_details_XGB(self, model):
        return self.get_details_GBM(model)

    def get_details_GLM(self, model):
        metric = {}
        loss_iter = {}
        act_vs_pre = {}

        metric = self.calculate_metric(model.model_performance())
        loss_iter = self.calculate_loss_vs_time_glm(model.score_history())
        act_vs_pre = self.calculate_act_vs_pre(model, self.h2otest, self.y_variable)

        model_details = {}
        graph = {}
        model_details['Metric'] = metric
        graph['Loss vs Iterations'] = loss_iter
        graph['Actual vs Predictions'] = act_vs_pre
        model_details['graph'] = graph

        return model_details

    def get_details_DL(self, model):
        model_details = self.get_details_ST(model)

        loss_iter = self.calculate_loss_vs_time(model.score_history())
        model_details['graph']['Loss vs Iterations'] = loss_iter

        return model_details

    def get_details_ST(self, model):
        metric = {}
        loss_iter = {}
        act_vs_pre = {}

        metric = self.calculate_metric(model.model_performance())
        act_vs_pre = self.calculate_act_vs_pre(model, self.h2otest, self.y_variable)

        model_details = {}
        graph = {}
        model_details['Metric'] = metric
        graph['Actual vs Predictions'] = act_vs_pre
        model_details['graph'] = graph

        return model_details

    def get_details_GBM(self, model):
        metric = {}
        variable_importance = {}
        loss_iter = {}
        act_vs_pre = {}

        metric = self.calculate_metric(model.model_performance())
        variable_importance = self.calculate_var_imp(model.varimp())
        loss_iter = self.calculate_loss_vs_time(model.score_history())
        act_vs_pre = self.calculate_act_vs_pre(model, self.h2otest, self.y_variable)

        model_details = {}
        graph = {}
        model_details['Metric'] = metric
        graph['Variable Importance'] = variable_importance
        graph['Loss vs Iterations'] = loss_iter
        graph['Actual vs Predictions'] = act_vs_pre
        model_details['graph'] = graph

        return model_details

    def calculate_metric(self, metrics):
        metric = {}
        metric['Mean Square Error'] = metrics.mse()
        metric['Root Mean Square Error'] = metrics.rmse()
        metric['Mean Absolute Error'] = metrics.mae()
        metric['Root Mean Square Algorithm Error'] = metrics.rmsle()
        metric['Mean Residual Deviance'] = metrics.mean_residual_deviance()

        return metric

    def calculate_var_imp(self, vimp):
        variable_importance = {}
        k = []
        v = []
        for v in vimp:
            variable_importance[v[0]] = v[2]
        variable_importance = {k: v for k, v in sorted(variable_importance.items(), key=lambda x: x[1])}
        #     print(v[0])
        #     print(v[2])
        #     k.append(v[0])
        #     v.append(v[2])
        #
        # list1 = np.array(v)
        # list2 = np.array(k)
        # idx   = np.argsort(list1)
        #
        # v = np.array(list1)[idx]
        # k = np.array(list2)[idx]
        #
        # for i in range(len(k)):
        #     variable_importance[k[i]]= float(v[i])

        variable_importance['graph_type'] = 'bar_chart'
        return variable_importance

    def calculate_loss_vs_time(self, scor_his):
        loss_vs_iter = {}
        loss_vs_iter['y_training'] = list(np.array(scor_his['training_rmse'])[1:])
        loss_vs_iter['y_validation'] = list(np.array(scor_his['validation_rmse'])[1:])
        loss_vs_iter['x'] = [i for i in range(len(loss_vs_iter['y_validation']))]
        loss_vs_iter['graph_type'] = "line"
        return loss_vs_iter

    def calculate_loss_vs_time_glm(self, scor_his):
        loss_vs_iter = {}
        loss_vs_iter['y_training'] = list(np.array(scor_his['deviance_train'])[1:])
        loss_vs_iter['y_validation'] = list(np.array(scor_his['deviance_train'])[1:])
        loss_vs_iter['x'] = [i for i in range(len(loss_vs_iter['y_validation']))]
        loss_vs_iter['graph_type'] = "line"
        return loss_vs_iter

    def calculate_act_vs_pre(self, model, df, y_variable):

        # df is the h20
        nrows= df.nrows
        if nrows>300:
            df2=df[:300,:]
        else:
            df2 = df
        act = np.array(df2[y_variable].as_data_frame())[:, 0]
        act = np.nan_to_num(act)
        hf = df2
        p = model.predict(hf)
        pre = np.array(p['predict'].as_data_frame())[:, 0]

        act_vs_pre = {}
        actual = list(act)
        new_actual = []
        for a in actual:
            if a == "Nan":
                new_actual.append(float(0))
            else:
                new_actual.append(float(a))
        new_pre = []
        for p in pre:
            new_pre.append(float(p))

        act_vs_pre['y_actual'] = new_actual
        act_vs_pre['y_prediction'] = new_pre
        act_vs_pre['x'] = [i for i in range(len(act_vs_pre['y_prediction']))]
        act_vs_pre['graph_type'] = "line"
        return act_vs_pre

    def predict(self, model, df, df_before=None, time=None):
        df2 = df.toPandas()
        hf = h2o.H2OFrame(df2)
        p = model.predict(hf)
        p_data_frame = p.as_data_frame()
        for c in p_data_frame.columns:
            df2[c] = p_data_frame[c]

        lst = list(np.array(p_data_frame['predict']))
        new_lst = []
        for l in lst:
            new_lst.append(float(l))
        return new_lst, df2

    def get_x_from_frame(self, df, time_var=None):
        if time_var == None:
            x = []
            for i in range(df.count()):
                x.append(i)
            return x
        else:
            lst = list(np.array(df.select(time_var).toPandas()))
            new_lst = []
            for l in lst:
                new_lst.append(str(l))
            return new_lst

    def x_and_y_graph(self, x, y):
        return {"x": x, "y": y}

    def save_model(self, ID=None, key=None, model=None, bucket='mlttronsbucket1', local=False):
        if model == None:
            print("Model not given")
            return False

        if ID == None or key == None:
            print("ID and key not given")
            return False

        path = 'model/' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3].replace(" ", "")

        # save the model
        try:
            model_path = h2o.save_model(model=model, path='/tmp/' + path, force=True)
            # model_path = h2o.save_model(model=model, path=os.path.join(settings.BASE_DIR, 'spark', path), force=True)

            session = boto3.Session(
                aws_access_key_id=ID,
                aws_secret_access_key=key,
            )

            s3 = session.client('s3')
            res = s3.upload_file(model_path, bucket, path)
            print(res)
            os.remove(model_path)
            return path

            # print("model path", model_path)
            # return  model_path

        except Exception as e:
            print("exception in model path")
            print(e)
            return False

    def load_model(self, ID=None, key=None, bucket='mlttronsbucket1', path=None):
        if path == None:
            print("Path not given")
            return False

        if ID == None or key == None:
            print("ID and key not given")
            return False

        try:
            session = boto3.Session(
                aws_access_key_id=ID,
                aws_secret_access_key=key,
            )

            s3 = session.resource('s3')
            local_path = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
            res = s3.Bucket(bucket).download_file(path, '/tmp/' + local_path)
            print(res)

            h2o.init(max_mem_size="6g")

            model = h2o.load_model('/tmp/' + local_path)
            # model = h2o.load_model(os.path.join(settings.BASE_DIR, 'spark', path))

            os.remove('/tmp/' + local_path)
            return model
        except Exception as e:
            print(e)
            return False

    # def retrain(self, model=None, model_name=None, df=None, variables=None, y_variable=None):
    #
    #     from h2o.estimators.gbm import H2OGradientBoostingEstimator
    #     from h2o.estimators.deeplearning import H2ODeepLearningEstimator
    #     from h2o.estimators.random_forest import H2ORandomForestEstimator
    #
    #     if model_name == None:
    #         print("Please provide model_name.")
    #         return False
    #
    #     predictors = []
    #     for v in variables:
    #         predictors.append(v[0])
    #
    #     if model_name == 'Gradient Boosting Machine':
    #
    #         # build and train model with 5 additional trees:
    #         model_continued = H2OGradientBoostingEstimator(checkpoint=model.model_id, ntrees=model.ntrees + 5)
    #
    #         df2 = df.toPandas()
    #         train = h2o.H2OFrame(df2)
    #         model_continued.train(x=predictors, y=y_variable, training_frame=train, validation_frame=valid)
    #
    #         return model_continued
    #
    #     elif model_name == 'Distributed Random Forest':
    #
    #         # build and train model with 5 additional trees:
    #         model_continued = H2ORandomForestEstimator(checkpoint=model.model_id, ntrees=model.ntrees + 5)
    #
    #         df2 = df.toPandas()
    #         train = h2o.H2OFrame(df2)
    #         model_continued.train(x=predictors, y=y_variable, training_frame=train, validation_frame=valid)
    #
    #         return model_continued
    #
    #     elif model_name == 'Deep Learning':
    #
    #         # build and train model with 5 additional trees:
    #         model_continued = H2ODeepLearningEstimator(checkpoint=model.model_id, epochs=model.epochs + 15)
    #
    #         df2 = df.toPandas()
    #         train = h2o.H2OFrame(df2)
    #         model_continued.train(x=predictors, y=y_variable, training_frame=train, validation_frame=valid)
    #
    #         return model_continued
    #
    #     else:
    #         print("Retraing not possible. Please run automl again.")
    #         return False
