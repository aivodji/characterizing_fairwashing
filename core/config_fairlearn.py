

def get_data(id):
    dataset, decision, maj_grp, min_grp = None, None, None, None

    if id==1:
        dataset = "adult_income"
        decision = "income"
        maj_grp = "gender_Male"
        min_grp = "gender_Female"

    if id==2:
        dataset = "compas"
        decision = "low_risk"
        maj_grp = "race_Caucasian"
        min_grp = "race_African-American"
        

    if id==3:
        dataset = "default_credit"
        decision = "good_credit"
        maj_grp = "SEX_Male"
        min_grp = "SEX_Female"
        
        
    if id==4:
        dataset = "marketing"
        decision = "subscribed"
        maj_grp = "age_age:30-60"
        min_grp = "age_age:not30-60"
        
    

    return dataset, decision, maj_grp, min_grp

def get_metric(metric):
    metrics = {
    1: "statistical_parity",
    2 : "predictive_parity",
    3 : "predictive_equality",
    4 : "equal_opportunity",
    5 : "equalized_odds",
    6 : "conditional_use_accuracy_equality"
    }
    return metrics[metric]

