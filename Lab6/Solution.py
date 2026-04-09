# Import required library
import pandas as pd

# Load the data from Excel file
data = pd.read_excel(r'C:\Users\Student\Downloads\ML_Lab6_data.xlsx')

# Extract features and labels
emails = data['Email'].tolist()
free = data['Free'].tolist()
win = data['Win'].tolist()
money = data['Money'].tolist()
spam = data['Spam'].tolist()

# Total number of emails
total_emails = len(spam)

# a) Compute Prior Probabilities
spam_yes_count = spam.count("Yes")
spam_no_count = spam.count("No")

p_spam_yes = spam_yes_count / total_emails
p_spam_no = spam_no_count / total_emails

print("a) Prior Probabilities:")
print(f"P(Spam = Yes) = {spam_yes_count}/{total_emails} = {p_spam_yes}")
print(f"P(Spam = No) = {spam_no_count}/{total_emails} = {p_spam_no}")
print()

# b) Calculate conditional probabilities
# Find indices where spam = Yes
spam_yes_indices = [i for i in range(total_emails) if spam[i] == "Yes"]
spam_yes_count = len(spam_yes_indices)

# P(Free = Yes | Spam = Yes)
free_yes_in_spam_yes = 0
for i in spam_yes_indices:
    if free[i] == "Yes":
        free_yes_in_spam_yes += 1

p_free_yes_given_spam_yes = free_yes_in_spam_yes / spam_yes_count

# P(Money = Yes | Spam = Yes)
money_yes_in_spam_yes = 0
for i in spam_yes_indices:
    if money[i] == "Yes":
        money_yes_in_spam_yes += 1

p_money_yes_given_spam_yes = money_yes_in_spam_yes / spam_yes_count

print("b) Conditional Probabilities:")
print(f"P(Free = Yes | Spam = Yes) = {free_yes_in_spam_yes}/{spam_yes_count} = {p_free_yes_given_spam_yes}")
print(f"P(Money = Yes | Spam = Yes) = {money_yes_in_spam_yes}/{spam_yes_count} = {p_money_yes_given_spam_yes}")
print()

# c) Compute P(Spam | Free=Yes, Win=Yes, Money=No)
# First, find all emails with Free=Yes, Win=Yes, Money=No
target_emails = []
for i in range(total_emails):
    if free[i] == "Yes" and win[i] == "Yes" and money[i] == "No":
        target_emails.append(i)

print("c) Computing P(Spam | Free=Yes, Win=Yes, Money=No):")
print(f"Emails with Free=Yes, Win=Yes, Money=No: {target_emails}")

# Calculate for Spam = Yes
# P(Spam = Yes)
p_spam_yes = spam_yes_count / total_emails

# P(Free=Yes | Spam=Yes)
p_free_yes_given_spam_yes = 0
spam_yes_indices = [i for i in range(total_emails) if spam[i] == "Yes"]
for i in spam_yes_indices:
    if free[i] == "Yes":
        p_free_yes_given_spam_yes += 1
p_free_yes_given_spam_yes = p_free_yes_given_spam_yes / len(spam_yes_indices)

# P(Win=Yes | Spam=Yes)
p_win_yes_given_spam_yes = 0
for i in spam_yes_indices:
    if win[i] == "Yes":
        p_win_yes_given_spam_yes += 1
p_win_yes_given_spam_yes = p_win_yes_given_spam_yes / len(spam_yes_indices)

# P(Money=No | Spam=Yes)
p_money_no_given_spam_yes = 0
for i in spam_yes_indices:
    if money[i] == "No":
        p_money_no_given_spam_yes += 1
p_money_no_given_spam_yes = p_money_no_given_spam_yes / len(spam_yes_indices)

# Calculate for Spam = No
# P(Spam = No)
spam_no_indices = [i for i in range(total_emails) if spam[i] == "No"]
p_spam_no = len(spam_no_indices) / total_emails

# P(Free=Yes | Spam=No)
p_free_yes_given_spam_no = 0
for i in spam_no_indices:
    if free[i] == "Yes":
        p_free_yes_given_spam_no += 1
p_free_yes_given_spam_no = p_free_yes_given_spam_no / len(spam_no_indices) if len(spam_no_indices) > 0 else 0

# P(Win=Yes | Spam=No)
p_win_yes_given_spam_no = 0
for i in spam_no_indices:
    if win[i] == "Yes":
        p_win_yes_given_spam_no += 1
p_win_yes_given_spam_no = p_win_yes_given_spam_no / len(spam_no_indices) if len(spam_no_indices) > 0 else 0

# P(Money=No | Spam=No)
p_money_no_given_spam_no = 0
for i in spam_no_indices:
    if money[i] == "No":
        p_money_no_given_spam_no += 1
p_money_no_given_spam_no = p_money_no_given_spam_no / len(spam_no_indices) if len(spam_no_indices) > 0 else 0

# Calculate numerator for Spam=Yes and Spam=No
numerator_spam_yes = p_spam_yes * p_free_yes_given_spam_yes * p_win_yes_given_spam_yes * p_money_no_given_spam_yes
numerator_spam_no = p_spam_no * p_free_yes_given_spam_no * p_win_yes_given_spam_no * p_money_no_given_spam_no

# Calculate denominator (evidence)
evidence = numerator_spam_yes + numerator_spam_no

# Calculate final probabilities
if evidence > 0:
    p_spam_yes_given_features = numerator_spam_yes / evidence
    p_spam_no_given_features = numerator_spam_no / evidence
else:
    p_spam_yes_given_features = 0
    p_spam_no_given_features = 0

print(f"\nP(Spam=Yes | Free=Yes, Win=Yes, Money=No) = {p_spam_yes_given_features:.4f}")
print(f"P(Spam=No | Free=Yes, Win=Yes, Money=No) = {p_spam_no_given_features:.4f}")
