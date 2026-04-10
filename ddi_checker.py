# import csv
# # Function: Show Severity Level
# def show_severity(severity):
#     if severity.lower() == "high":
#         print(" HIGH RISK")
#     elif severity.lower() == "moderate":
#         print(" MODERATE RISK")
#     else:
#         print(" LOW RISK")

# # Function: Check Drug Interactio
# def check_interaction(drug1, drug2):
#     if drug1 == drug2:
#         print("⚠ You entered the same drug!")
#         return

#     found = False

#     with open("drug_interactions.csv", "r") as file:
#         reader = csv.DictReader(file)

#         for row in reader:
#             d1 = row["Drug1"].lower()
#             d2 = row["Drug2"].lower()

#             if (drug1 == d1 and drug2 == d2) or (drug1 == d2 and drug2 == d1):
#                 print("\n⚠ Interaction Found!")
#                 show_severity(row["Severity"])
#                 print("Description:", row["Description"])
#                 found = True
#                 break

#     if not found:
#         print("\n No interaction found between these drugs.")

# # Function: Show All Drug
# def show_all_drugs():
#     drugs = set()

#     with open("drug_interactions.csv", "r") as file:
#         reader = csv.DictReader(file)

#         for row in reader:
#             drugs.add(row["Drug1"])
#             drugs.add(row["Drug2"])

#     print("\n Available Drugs in Database:")
#     for d in sorted(drugs):
#         print("-", d)

# # MAIN PROGRAM (Menu System)
# while True:
#     print("\n==============================")
#     print(" DRUG INTERACTION SYSTEM")
#     print("==============================")
#     print("1. Check Drug Interaction")
#     print("2. Show All Drugs")
#     print("3. Exit")

#     choice = input("Enter your choice: ")

#     if choice == "1":
#         d1 = input("Enter first drug: ").lower()
#         d2 = input("Enter second drug: ").lower()
#         check_interaction(d1, d2)

#     elif choice == "2":
#         show_all_drugs()

#     elif choice == "3":
#         print(" Exiting system... Stay safe!")
#         break

#     else:
#         print(" Invalid choice! Try again.")


























import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# TRAIN ML MODEL 

df = pd.read_csv("drug_interactions.csv")

X = df["Description"]
y = df["Severity"]

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_vector, y)

print(" ML Model Ready!")

# PREDICT FUNCTION

def predict_severity(description):
    desc_vector = vectorizer.transform([description])
    prediction = model.predict(desc_vector)
    return prediction[0]

# SHOW SEVERITY
def show_severity(severity):
    if severity.lower() == "high":
        print(" HIGH RISK")
    elif severity.lower() == "moderate":
        print(" MODERATE RISK")
    else:
        print(" LOW RISK")

# CHECK INTERACTION
def check_interaction(drug1, drug2):
    if drug1 == drug2:
        print("⚠ Same drug entered!")
        return

    found = False

    with open("drug_interactions.csv", "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            d1 = row["Drug1"].lower()
            d2 = row["Drug2"].lower()

            if (drug1 == d1 and drug2 == d2) or (drug1 == d2 and drug2 == d1):

                print("\n⚠ Interaction Found!")
                
                #  ML Prediction instead of direct severity
                predicted = predict_severity(row["Description"])
                
                show_severity(predicted)
                print("Description:", row["Description"])

                found = True
                break

    if not found:
        print("\n No interaction found in dataset.")
        print(" Model is ready to learn ")

        # Use ML even if not found
        test_desc = input("Enter interaction description: ")
        predicted = predict_severity(test_desc)

        show_severity(predicted)
        
# SHOW ALL DRUGS

def show_all_drugs():
    drugs = set()

    with open("drug_interactions.csv", "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            drugs.add(row["Drug1"])
            drugs.add(row["Drug2"])

    print("\n Available Drugs:")
    for d in sorted(drugs):
        print("-", d)


# MENU SYSTEM

while True:
    print("\n==============================")
    print("AI DRUG INTERACTION SYSTEM")
    print("==============================")
    print("1. Check Drug Interaction")
    print("2. Show All Drugs")
    print("3. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        d1 = input("Enter first drug: ").lower()
        d2 = input("Enter second drug: ").lower()
        check_interaction(d1, d2)

    elif choice == "2":
        show_all_drugs()

    elif choice == "3":
        print("Exiting...")
        break

    else:
        print(" Invalid choice!")
        
        
        
        
        
        
       
    






















































































































































































































































































# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier

# # Load dataset
# df = pd.read_csv("drug_interactions.csv")

# # Train model
# X = df["Description"]
# y = df["Severity"]

# vectorizer = TfidfVectorizer()
# X_vector = vectorizer.fit_transform(X)

# model = RandomForestClassifier(n_estimators=100)
# model.fit(X_vector, y)

# print("✅ ML Model Ready!")

# # MAIN FUNCTION (IMPORTANT)
# def predict_interaction(drug1, drug2):

#     drug1 = drug1.lower()
#     drug2 = drug2.lower()

#     for _, row in df.iterrows():
#         d1 = row["Drug1"].lower()
#         d2 = row["Drug2"].lower()

#         if (drug1 == d1 and drug2 == d2) or (drug1 == d2 and drug2 == d1):
            
#             predicted = model.predict(vectorizer.transform([row["Description"]]))
            
#             return predicted[0] + " | " + row["Description"]

#     # If not found
#     return "No interaction found in dataset"