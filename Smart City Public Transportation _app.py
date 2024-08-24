from flask import Flask, render_template, request
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

df=pd.read_csv("traffic_data.csv")
df.head()
columns_to_remove = ['max_speed', 'average_daily_car_traffic', 'average_daily_bike_traffic']
df = df.drop(columns=columns_to_remove)

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

categorical_columns = df.select_dtypes(include='object').columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))
numeric_data = df.select_dtypes(include=['float64', 'int64']).dropna()
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)
gmm = GaussianMixture(n_components=3, random_state=42)
df['gmm_cluster'] = gmm.fit_predict(scaled_data)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        input_data = {}
        for column in ['_id', 'id', 'device_id', 'record_oid', 'speed_limit', 'median_speed', 'percent_over_limit',
                       'speed85_percent', 'speed95_percent', 'longitude', 'latitude', 'council_district', 'ward',
                       'tract', 'public_works_division', 'pli_division', 'police_zone']:
            input_data[column] = float(request.form.get(column))

        # Use the trained GMM model to predict the cluster
        input_data_df = pd.DataFrame([input_data])  # Convert input data to DataFrame
        input_data_scaled = scaler.transform(input_data_df[numeric_data.columns])  # Use the same set of features
        predicted_cluster = gmm.predict(input_data_scaled)[0]

        # Filter data for the predicted cluster
        cluster_data = df[df['gmm_cluster'] == predicted_cluster]

        # Plot the graph
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='longitude', y='latitude', hue='gmm_cluster', data=df, palette='viridis', legend='full')
        sns.scatterplot(x='longitude', y='latitude', hue='gmm_cluster', data=cluster_data, palette='Reds', legend='full', s=100)
        plt.title('Gaussian Mixture Model Clustering')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')

        # Save the plot as an image
        plot_path = 'static/cluster_plot.png'
        plt.savefig(plot_path)
        plt.close()  # Close the plot to prevent it from being displayed in the response

        return render_template('index.html', predicted_cluster=predicted_cluster, plot_path=plot_path)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)