import streamlit as st
import pickle
import random
import plotly.graph_objects as go
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest

# Load the machine learning model
model_up = pickle.load(open('fraud_model.pkl', 'rb'))

# Function to predict fraud
def predict_fraud(transaction_type, amount, old_balance):
    # Map transaction_type to numeric values
    transaction_type_mapping = {'Transfer': 0, 'Payment': 1, 'Debit': 2, 'Cash_In': 3, 'Cash_Out': 4}
    numeric_transaction_type = transaction_type_mapping.get(transaction_type, -1)  # -1 if not found

    if numeric_transaction_type == -1:
        st.warning("Invalid transaction type. Please select a valid transaction type.")
        return None

    new_balance = old_balance - amount
    result = model_up.predict_proba([[numeric_transaction_type, amount, old_balance, new_balance]])[:, 1][0]
    return result

# Function to simulate new transaction data alternately as 1 fraud followed by 1 non-fraud
def simulate_new_transaction(fraud_next):
    if fraud_next:
        # Generate a fraud transaction
        transaction_type = 'Transfer'
        amount = round(random.uniform(10, 1000), 2)
        old_balance = round(random.uniform(1000, 10000), 2)
        old_balance -= 1000  # Simulate a fraud transaction
    else:
        # Generate a non-fraud transaction
        transaction_type = random.choice(['Payment', 'Debit', 'Cash_In', 'Cash_Out'])
        amount = round(random.uniform(10, 1000), 2)
        old_balance = round(random.uniform(1000, 10000), 2)

    return transaction_type, amount, old_balance

# Streamlit App
def main():
    st.title("Real-Time Fraud Detection with Machine Learning")

    # Transaction inputs
    amount = st.number_input("Enter transaction amount")
    old_bal = st.number_input("Enter old balance")

    # Radio button for transaction type
    transaction_type = st.radio(
        "Select transaction type:",
        ('Transfer', 'Payment', 'Debit', 'Cash_In', 'Cash_Out')
    )

    # Create an empty list for transaction data
    transaction_data = []

    # Create an empty Plotly figure
    fig = go.Figure()

    # Initialize fraud_next to True to start with a fraud transaction
    fraud_next = True

    # Initialize confusion matrix variables
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # Prediction button
    if st.button('Predict Fraud'):
        predicted_probability = predict_fraud(transaction_type, amount, old_bal)

        if predicted_probability is not None:
            # Display transaction result
            if predicted_probability > 0.5:
                st.warning("This transaction is likely a Fraud with probability {:.2f}%".format(predicted_probability * 100))
                st.error("ALERT: Potential Fraud Detected!")
            else:
                st.success("This transaction is likely Not a Fraud with probability {:.2f}%".format((1 - predicted_probability) * 100))

    if st.button('Start Real-Time Monitoring'):
        try:
            # Initialize summary variables
            total_transactions = 0
            fraud_transactions = 0
            not_fraud_transactions = 0

            # Create placeholders for summary text in the sidebar
            st.sidebar.header("Summary Statistics")
            total_text = st.sidebar.text("Total Transactions: 0")
            fraud_text = st.sidebar.text("Fraud Transactions: 0")
            not_fraud_text = st.sidebar.text("Not Fraud Transactions: 0")

            # Placeholder for precision, recall, and F1-score
            st.sidebar.text("Model Performance:")
            precision_text = st.sidebar.text("Precision: 0.00")
            recall_text = st.sidebar.text("Recall: 0.00")
            f1_text = st.sidebar.text("F1-Score: 0.00")

            # Placeholder for Cumulative Fraud Rate
            cumulative_fraud_text = st.sidebar.text("Cumulative Fraud Rate: 0.00%")

            while True:
                # Simulate new transaction
                new_transaction_type, new_amount, new_old_balance = simulate_new_transaction(fraud_next)
                predicted_probability = predict_fraud(new_transaction_type, new_amount, new_old_balance)

                if predicted_probability is not None:
                    # Update confusion matrix and calculate metrics
                    if predicted_probability > 0.5:  # Predicted fraud
                        if fraud_next:  # True fraud
                            true_positive += 1
                        else:  # False positive
                            false_positive += 1
                        fraud_transactions += 1
                    else:  # Predicted not fraud
                        if fraud_next:  # False negative
                            false_negative += 1
                        else:  # True negative
                            true_negative += 1
                        not_fraud_transactions += 1

                    total_transactions += 1
                    fraud_next = not fraud_next

                    # Calculate precision, recall, and F1-score
                    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
                    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    # Update the summary text in the sidebar
                    total_text.text(f"Total Transactions: {total_transactions}")
                    fraud_text.text(f"Fraud Transactions: {fraud_transactions}")
                    not_fraud_text.text(f"Not Fraud Transactions: {not_fraud_transactions}")
                    precision_text.text(f"Precision: {precision:.2f}")
                    recall_text.text(f"Recall: {recall:.2f}")
                    f1_text.text(f"F1-Score: {f1:.2f}")

                    # Cumulative Fraud Rate
                    cumulative_fraud_rate = (fraud_transactions / total_transactions) * 100 if total_transactions > 0 else 0
                    cumulative_fraud_text.text(f"Cumulative Fraud Rate: {cumulative_fraud_rate:.2f}%")

                    # Append data to the list
                    transaction_data.append({"Transaction Type": new_transaction_type, "Amount": new_amount, "Old Balance": new_old_balance, "Predicted Probability": predicted_probability})

                    # Convert the list to a DataFrame
                    transaction_df = pd.DataFrame(transaction_data)

                    # Real-time table display of transactions
                    st.dataframe(transaction_df[['Transaction Type', 'Amount', 'Old Balance', 'Predicted Probability']])

                    # Update the Plotly figure with new data
                    fig.add_trace(go.Scatter(x=transaction_df.index, y=transaction_df["Predicted Probability"], mode='lines+markers', name='Predicted Probability'))

                    # Update the Plotly layout
                    fig.update_layout(title="Real-Time Predicted Probability",
                                      xaxis_title="Transaction",
                                      yaxis_title="Predicted Probability")

                    # Display the updated Plotly figure
                    st.plotly_chart(fig)

                time.sleep(2)  # Simulate a delay between transactions

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Feature importance (dummy example for illustration)
    # Assuming feature importance exists in your model
    st.sidebar.header("Feature Importance")
    feature_importance = [0.2, 0.3, 0.1, 0.4]  # Dummy data
    feature_names = ['Transaction Type', 'Amount', 'Old Balance', 'New Balance']
    fig_feature = go.Figure([go.Bar(x=feature_names, y=feature_importance)])
    st.sidebar.plotly_chart(fig_feature)

if __name__ == '__main__':
    main()
