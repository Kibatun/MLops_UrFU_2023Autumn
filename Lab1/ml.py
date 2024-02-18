from clearml import PipelineDecorator
from clearml import Task, Logger
from sklearn.metrics import precision_recall_fscore_support


@PipelineDecorator.component(return_values=['train, test'], execution_queue="default")
def data_load():
    import pandas as pd
    import numpy as np

    train=pd.read_csv('train1.csv')
    test=pd.read_csv('test1.csv')
    return train,test


@PipelineDecorator.component(return_values=['train_without,target'], execution_queue="default")
def data_drop(train):
    import pandas as pd
    import numpy as np

    target = train['target']
    train_without = train.drop(columns=['target', 'row_id'])
        
    return train_without, target


@PipelineDecorator.component(return_values=['train_pca'], execution_queue="default")
def pca(train_without):
    import numpy as np
    from sklearn.decomposition import PCA
    

    pca = PCA(n_components=10)
    pca.fit(train_without)
    train_pca = pca.transform(train_without)
    
    return train_pca


@PipelineDecorator.component(return_values=['X_train, X_test, y_train, y_test'], execution_queue="default")
def split(train_pca,target):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train_pca, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


@PipelineDecorator.component(return_values=['model'], execution_queue="default")
def model_train(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    return model


@PipelineDecorator.component(return_values=['score'], execution_queue="default")
def prediction(model, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred),2)
    precision = round(precision_score(y_test, y_pred, average = "weighted"),2)
    recall = round(recall_score(y_test, y_pred,average = "weighted"),2)
    f1 = round(f1_score(y_test, y_pred,average = "weighted"),2)
    return accuracy, precision, recall, f1
    
    
@PipelineDecorator.component(return_values=['report'], execution_queue="default")
def classification_report_df(model, X_test, y_test):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support
    y_pred = model.predict(X_test)
    clf_rep = precision_recall_fscore_support(y_test, y_pred)
    classes = list(set(y_test))
    avgs=[]
    totalsum = np.sum(clf_rep[3])
    for i in range(0,3):
        avgs.append(np.sum(clf_rep[i]*clf_rep[3])/totalsum)

    avgs.append(totalsum)
    mylist = [list(x) for x in clf_rep]
    clf_rep_all = [x + [y] for x,y in zip(mylist,avgs)]
    indices = list(classes) +['avg/total']
    out_dict = {
                 "precision" :clf_rep_all[0]
                ,"recall" : clf_rep_all[1]
                ,"f1-score" : clf_rep_all[2]
                ,"support" : clf_rep_all[3]
                }
    out_df = pd.DataFrame(out_dict, index = indices)
    out_df[["precision","recall","f1-score"]]= out_df[["precision","recall","f1-score"]].apply(lambda x: round(x,2))
    return out_df


@PipelineDecorator.pipeline(name='test', project='kaggle_project', version='0.0.1')
def executing_pipeline():
    from matplotlib import pyplot as plt 
    import seaborn as sns 
    task = Task.create(project_name="kaggle_project", task_name="knn")
    logger = task.get_logger()
    train, test = data_load()
    train_without, target = data_drop(train)
    sns.countplot(x=target)
    plt.title('Targets value count')
    plt.xlabel('Count')
    plt.ylabel('Bacteria')
    task.logger.report_matplotlib_figure(
    title='Targets value count', series="Just a plot", iteration=0, figure=plt)
 
    train_pca = pca(train_without)
    X_train, X_test, y_train, y_test = split(train_pca,target)
    model = model_train(X_train, y_train)
    preds = prediction(model, X_test, y_test)
    logger.report_single_value(name="accuracy", value=preds[0])
    logger.report_single_value(name="precision", value=preds[1])
    logger.report_single_value(name="recall", value=preds[2])
    logger.report_single_value(name="f1", value=preds[3])
    logger.report_table(title='Classification report',series='pandas DataFrame',iteration=0,table_plot=classification_report_df(model, X_test, y_test))

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    executing_pipeline()







