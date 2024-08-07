# app.py

from flask import Flask, render_template, request
from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix





app = Flask(__name__)

# Function to download S&P 500 data and perform linear regression
def lin(test_size, sample_size, days_adjust=7, features=['Open', 'High', 'Low', 'Volume', 'Close']):
    today = date.today()
    data = yf.download('^GSPC', start='2000-01-01', end=today)
    
    print(features)

    # Create Next_Week_Close column
    data["Next_Week_Close"] = 0
    adjusted_data = data[:-days_adjust]

    for index in range(len(adjusted_data)):
        adjusted_data["Next_Week_Close"][index] = data["Close"][index + days_adjust]

    old_data = adjusted_data

    # Adjusting sample size
    adjust = int(sample_size * len(adjusted_data))
    adjusted_data = adjusted_data[:adjust]

    # Feature selection
    X = adjusted_data[features].values
    y = adjusted_data['Next_Week_Close'].values

    old_X = old_data[features].values
    old_y = old_data['Next_Week_Close'].values

    # Creating train and test using test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize Linear Regression model
    model = LinearRegression()

    # Check for NaN and Infinity values in X_train
    nan_indices = np.isnan(X_train).any(axis=1)
    inf_indices = np.isinf(X_train).any(axis=1)

    # Check for NaN and Infinity values in y_train
    nan_indices_y = np.isnan(y_train)
    inf_indices_y = np.isinf(y_train)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plot the actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(old_y, label='Actual Prices', color='blue')
    plt.plot(model.predict(old_X), label='Predicted Prices', color='red', linestyle=':')
    plt.title('Actual vs. Predicted Prices for S&P 500 (^GSPC)')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Convert plot to base64 for embedding in HTML
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return mse, r2, plot_url, old_y




# Function to perform K-Means clustering
def kmeans(test_size, sample_size, num_clusters, features=['Open', 'High', 'Low', 'Volume', 'Close']):
    today = date.today()
    data = yf.download('^GSPC', start='2000-01-01', end=today)

    # Create Next_Week_Close column
    data["Next_Week_Close"] = 0
    adjusted_data = data[:-1]

    for index in range(len(adjusted_data)):
        adjusted_data["Next_Week_Close"][index] = data["Close"][index + 1]

    old_data = adjusted_data

    # Adjusting sample size
    adjust = int(sample_size * len(adjusted_data))
    adjusted_data = adjusted_data[:adjust]

    # Feature selection
    X = adjusted_data[features].values

    # Initialize K-Means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Fit K-Means model
    kmeans.fit(X)

    # Get cluster labels
    labels = kmeans.labels_

    # Number of samples in each cluster
    unique, counts = np.unique(labels, return_counts=True)

    # Plotting the clusters
    plt.figure(figsize=(12, 6))
    plt.bar(unique, counts, align='center', alpha=0.5)
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Samples in Clusters')
    plt.grid(True)

    # Convert plot to base64 for embedding in HTML
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url





def svm_classification(test_size, sample_size, features, data=pd.read_csv("Data/IRIS.csv")):
    # Adjusting sample size
    adjust = int(sample_size * len(data))
    print("Sample Size: ", adjust)
    adjusted_data = data.sample(n=adjust, random_state=42)
    
    # Convert species to numeric labels
    le = LabelEncoder()
    adjusted_data['species'] = le.fit_transform(adjusted_data['species'])
    
    # Feature selection
    X = adjusted_data[features].values
    y = adjusted_data["species"].values
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print("Training set size: ", len(X_train))
    
    # Initialize SVM model
    model = SVC(kernel='linear', random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Round all values to one decimal place
    report_df = report_df.round(1)
    
    # Add a column for flower names
    report_df.insert(0, 'Flower Name', report_df.index)
    
    # Split the DataFrame into two parts
    # First part: all rows except the last two
    first_part_df = report_df.iloc[:-3]
    
    # Second part: the last two rows without column titles
    second_part_df = report_df.iloc[-3:].reset_index(drop=True)
    
    # Generate the confusion matrix plot
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Convert confusion matrix plot to base64 for embedding in HTML
    img_conf_matrix = BytesIO()
    plt.savefig(img_conf_matrix, format='png')
    img_conf_matrix.seek(0)
    conf_matrix_plot_url = base64.b64encode(img_conf_matrix.getvalue()).decode('utf-8')
    plt.close()
    
    # Generate the SVM decision boundary plot if there are exactly 2 features
    decision_boundary_plot_url = ""
    if len(features) == 2:
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 7))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='viridis')
        plt.title('SVM Decision Boundary')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.grid(True)
        
        # Convert decision boundary plot to base64 for embedding in HTML
        img_decision_boundary = BytesIO()
        plt.savefig(img_decision_boundary, format='png')
        img_decision_boundary.seek(0)
        decision_boundary_plot_url = base64.b64encode(img_decision_boundary.getvalue()).decode('utf-8')
        plt.close()
        
    
    # Convert DataFrames to HTML
    first_part_html = first_part_df.to_html(classes='table table-striped', index=False, border=0)
    second_part_html = second_part_df.to_html(classes='table table-striped', index=False, border=0, header=False)
    
    # Replace NaN with non-breaking space in the HTML
    first_part_html = first_part_html.replace('<td>nan</td>', '<td>&nbsp;</td>')
    second_part_html = second_part_html.replace('<td>nan</td>', '<td>&nbsp;</td>')
    
    # Wrap both HTML parts in a container div
    report_html = f'<div>{first_part_html}</div><div>{second_part_html}</div>'
    
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report_df.to_string())

    return conf_matrix_plot_url, decision_boundary_plot_url, report_html











def hierarchical_clustering_dendrogram(data, features, target='quality', num_clusters=5):
    # Compute the linkage matrix
    Z = linkage(data[features], method='ward')

    # Create clusters
    clusters = fcluster(Z, num_clusters, criterion='maxclust')
    data['cluster'] = clusters

    # Map cluster numbers to actual labels based on the most common label in each cluster
    cluster_labels = {}
    for cluster_num in range(1, num_clusters + 1):
        common_label = data[data['cluster'] == cluster_num][target].mode()[0]
        cluster_labels[cluster_num] = common_label

    # Determine the maximum distance for the dendrogram to color clusters
    max_d = Z[-(num_clusters-1), 2]

    # Plot the dendrogram with color_threshold
    plt.figure(figsize=(20, 10))
    dendrogram_data = dendrogram(Z, labels=data.index, leaf_rotation=90, leaf_font_size=10, color_threshold=max_d)
    plt.title(f'Hierarchical Clustering Dendrogram with {num_clusters} Clusters')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')

    # Plot horizontal lines to show the clusters
    plt.axhline(y=max_d, c='k')

    # Removing the text on the x-axis
    plt.xticks([])

    
    # Convert plot to base64 for embedding in HTML
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url



def decision_tree_classification(test_size, sample_size, features, data):
    adjust = int(sample_size / 100 * len(data))
    adjusted_data = data.sample(n=adjust, random_state=42)
    X = adjusted_data[features]
    y = adjusted_data['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report and convert it to a DataFrame
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Rename clusters
    labels = ["Disgusting", "Awful", "Mediocre", "Good", "Great"]
    if len(report_df) == len(labels):  # Ensure the length matches
        report_df.index = labels

    # Remove the first row
    report_df = report_df.drop(report_df.index[0])

    # Add a 'Cluster Grouping' column
    report_df.insert(0, 'Cluster Grouping', report_df.index)

    # Round all values to one decimal place
    report_df = report_df.round(1)
    
    for index in range(5):
        report_df["Cluster Grouping"][index] = labels[index]

    
    # Generate the confusion matrix plot
    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(4, 9))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Convert confusion matrix plot to base64 for embedding in HTML
    img_conf_matrix = BytesIO()
    plt.savefig(img_conf_matrix, format='png')
    img_conf_matrix.seek(0)
    conf_matrix_plot_url = base64.b64encode(img_conf_matrix.getvalue()).decode('utf-8')
    plt.close()

    # Generate the dendrogram plot
    dendrogram_plot_url = hierarchical_clustering_dendrogram(data, features, 'quality', num_clusters=5)
    
    # Convert report DataFrame to HTML
    
    report_html = report_df.to_html(classes='table table-striped', index=False)

    return conf_matrix_plot_url, dendrogram_plot_url, report_html





def logistic_weather_prediction(test_size, sample_size, data, features):
    # Adjusting sample size
    adjust = int(sample_size * len(data))
    print("Sample Size: ", adjust)
    sampled_data = data.sample(n=adjust, random_state=42)
    
    # Feature selection
    X = sampled_data[features].values
    y = sampled_data['Precip Type'].values
    
    # Creating train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print("Training set size: ", len(X_train))
    
    # Initialize Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Convert confusion matrix plot to base64 for embedding in HTML
    img_conf_matrix = BytesIO()
    plt.savefig(img_conf_matrix, format='png')
    img_conf_matrix.seek(0)
    conf_matrix_plot_url = base64.b64encode(img_conf_matrix.getvalue()).decode('utf-8')
    plt.close()
    
    # Generate classification report and convert it to a DataFrame
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Add a 'Cluster Grouping' column
    report_df.insert(0, 'Rain', report_df.index)

    # Round all values to one decimal place
    report_df = report_df.round(1)
    # Split the DataFrame into two parts
    # First part: all rows except the last two
    first_part_df = report_df.iloc[:-3]
    
    # Second part: the last two rows without column titles
    second_part_df = report_df.iloc[-3:].reset_index(drop=True)
    # Convert DataFrames to HTML
    first_part_html = first_part_df.to_html(classes='table table-striped', index=False, border=0)
    second_part_html = second_part_df.to_html(classes='table table-striped', index=False, border=0, header=False)
    
    # Replace NaN with non-breaking space in the HTML
    first_part_html = first_part_html.replace('<td>nan</td>', '<td>&nbsp;</td>')
    second_part_html = second_part_html.replace('<td>nan</td>', '<td>&nbsp;</td>')
    
    # Wrap both HTML parts in a container div
    report_html = f'<div>{first_part_html}</div><div>{second_part_html}</div>'
    report_html = report_df.to_html(classes='table table-striped', index=False)
    
    return conf_matrix_plot_url, report_html





# Home page with form to input parameters
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model = request.form['model']
        features = request.form.getlist('features')
        
        if model == 'linear_regression':
            
            test_size = float(request.form['test_size'])
            sample_size = float(request.form['sample_size'])
            days_adjust = int(request.form['days_adjust'])
            features = request.form.getlist('features')
            
            total_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            adjusted_features = [feat for feat in features if feat in total_features]
            
            mse, r2, plot_url, old_y = lin(test_size, sample_size, days_adjust, adjusted_features)
            return render_template('result.html', mse=mse, r2=r2, plot_url=plot_url, model=model)
        
        elif model == 'kmeans':
            test_size = float(request.form['test_size'])
            sample_size = float(request.form['sample_size'])
            num_clusters = int(request.form['num_clusters'])
            features = request.form.getlist('features')

            total_features = ['Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']
            adjusted_features = [feat for feat in features if feat in total_features]

            plot_url = kmeans(test_size, sample_size, num_clusters, adjusted_features)
            
            return render_template('result.html', plot_url=plot_url, model=model)
        
        elif model == 'SVM':
            test_size = float(request.form['test_size'])
            sample_size = float(request.form['sample_size'])/100
            features = request.form.getlist('features')
            
            total_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            adjusted_features = [feat for feat in features if feat in total_features]
            
            conf_matrix_plot_url, decision_boundary_plot_url, report_html = svm_classification(test_size, sample_size, adjusted_features)
            return render_template('result.html', conf_matrix_plot_url = conf_matrix_plot_url, decision_boundary_plot_url=decision_boundary_plot_url, report_html=report_html, model=model)
        
        elif model == 'Tree':
            test_size = float(request.form['test_size'])
            sample_size = float(request.form['sample_size'])
            features = request.form.getlist('features')
            
            adjusted_features = [feat for feat in features if feat in list(pd.read_csv("Data/WineQT_Clean.csv").columns)]
            
            conf_matrix_plot_url, dendrogram_plot_url, report_html = decision_tree_classification(test_size, sample_size, features=adjusted_features, data=pd.read_csv("Data/WineQT_Clean.csv"))

            return render_template('result.html', report = report_html, conf_matrix_plot_url=conf_matrix_plot_url, dendrogram_plot_url=dendrogram_plot_url, model="Tree")
        
        elif model == 'logistic':
            test_size = float(request.form['test_size'])
            sample_size = float(request.form['sample_size'])/100
            features = request.form.getlist('features')
            
            adjusted_features = [feat for feat in features if feat in list(pd.read_csv('Data/weatherHistory_Clean.csv').columns)]
            
            conf_matrix_plot_url, report_html = logistic_weather_prediction(test_size, sample_size, features=adjusted_features, data=pd.read_csv('Data/weatherHistory_Clean.csv'))

            return render_template('result.html', report = report_html, conf_matrix_plot_url=conf_matrix_plot_url, model=model)
        
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
