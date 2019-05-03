# Databricks notebook source
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# COMMAND ----------

training_data_df = pd.read_csv("/dbfs/FileStore/tables/all_stocks_5yr_amazon_train.csv", dtype=float)

# COMMAND ----------

X_training = training_data_df.drop('close', axis=1).values
Y_training = training_data_df[['close']].values

# COMMAND ----------

test_data_df = pd.read_csv("/dbfs/FileStore/tables/all_stocks_5yr_amazon_test.csv", dtype=float)

# COMMAND ----------

X_testing = test_data_df.drop('close', axis=1).values
Y_testing = test_data_df[['close']].values

# COMMAND ----------

X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# COMMAND ----------

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# COMMAND ----------

learning_rate = 0.001
training_epochs = 200

# COMMAND ----------

number_of_inputs = 4
number_of_outputs = 1

# COMMAND ----------

layer_1_nodes = 75
layer_2_nodes = 100
layer_3_nodes = 75

# COMMAND ----------

with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# COMMAND ----------

with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# COMMAND ----------

with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# COMMAND ----------

with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# COMMAND ----------

with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases

# COMMAND ----------

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# COMMAND ----------

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# COMMAND ----------

with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

saver = tf.train.Saver()

# COMMAND ----------

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  training_writer = tf.summary.FileWriter('./logs/training', session.graph)
  testing_writer = tf.summary.FileWriter('./logs/testing', session.graph)
  for epoch in range(training_epochs):
          session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
          # Every 5 training steps, log our progress
          if epoch % 5 == 0:
            training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_scaled_training, Y:Y_scaled_training})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)
            print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch, training_cost, testing_cost))
            final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
  print("Final Training cost: {}".format(final_training_cost))
  print("Final Testing cost: {}".format(final_testing_cost))
  Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_testing})
  Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)
  real_earnings = test_data_df['close'].values[0]
  predicted_earnings = Y_predicted[0][0]
  print("The actual close were ${}".format(real_earnings))
  print("Our neural network predicted close of ${}".format(predicted_earnings))

  save_path = saver.save(session, "logs/trained_model.ckpt")
  print("Model saved: {}".format(save_path))


# COMMAND ----------

import matplotlib.pyplot as plt

real_earningsAll = test_data_df['close']
fig, ax = plt.subplots()
ax.scatter(real_earningsAll, Y_predicted)
ax.plot([real_earningsAll.min(), real_earningsAll.max()], [real_earningsAll.min(),
real_earningsAll.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
display(fig)

# COMMAND ----------

import matplotlib.pyplot as plt

data_df = pd.read_csv("/dbfs/FileStore/tables/all_stocks_5yr_amazon_full.csv")

fig, ax = plt.subplots()
y_plot = data_df['close']
x_plot = data_df['date']
plt.plot(x_plot, y_plot, '-')

display(fig)
