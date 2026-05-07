def check_retraining(recall, drift):

    retrain = False

    if recall < 0.85:
        print("ALERT: Recall dropped below threshold")
        retrain = True

    if drift > 0.70:
        print("ALERT: Data drift exceeded threshold")
        retrain = True

    if retrain:
        print("Triggering automated retraining pipeline...")
    else:
        print("System stable")

# Simulated monitoring values
recall = 0.80
drift = 0.75

check_retraining(recall, drift)