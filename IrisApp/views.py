from django.shortcuts import render,redirect
from joblib import load
import numpy as np
import openai
openai.api_key = "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"
rfc_model_new = load("./savedModels/rfc_model_new.joblib")
wt_knn = load("./savedModels/wt_knn.joblib")
symptoms = load("./savedModels/symptoms.joblib")
classes = load("./savedModels/classes.joblib")

def diseasePrediction(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            symptoms_list = request.POST.getlist('options')
            def top_five(classes, model, input_list):
                mapping = {}
                pred_prob = model.predict_proba([input_list])

                for disease, prob in zip(classes, pred_prob[0]):
                    mapping[disease] = prob

                result = dict(sorted(mapping.items(), key=lambda x: x[1], reverse=True))

                # Get the keys of the dictionary as a list and slice the list
                sliced_keys = list(result.keys())[:5]

                # Create a new dictionary with the sliced keys
                sliced_dict = {k: result[k] for k in sliced_keys}

                # Print the sliced dictionary
                return sliced_dict
            def final_op(wt_knnl_dict, rfc_new_model_dict):
                final_op = {}
                for key, val in wt_knnl_dict.items():
                    final_op[key] = val

                for key, val in rfc_new_model_dict.items():
                    if key not in final_op:
                        final_op[key] = val
                    else:
                        v1 = final_op[key]
                        ans = (v1+val)/2
                        final_op[key] = ans
                return final_op
            new_data = symptoms_list
            input_list = []
            for i in symptoms:
                if i in new_data:
                    input_list.append(1)
                else:
                    input_list.append(0)
            input_list = np.array(input_list)
            rfc_new_model_dict = top_five(classes, rfc_model_new, input_list)
            wt_knnl_dict = top_five(classes, wt_knn, input_list)
            final = final_op(wt_knnl_dict, rfc_new_model_dict)
            information = []
            for disease in final.keys():
                completions = openai.Completion.create(
                engine="text-davinci-003",
                prompt="Tell me about " + disease,
                max_tokens=2048,
                n=1,
                temperature=0.5,
                )
                response = completions.choices[0].text
                information.append(response)
            
            return render(request, "predict.html", {"result": final , "symptoms":symptoms , 'my_info': information})
        return render(request, "predict.html", {"symptoms":symptoms})
    else:
        return redirect("landing")
        
def firstAid(request):
    if request.user.is_authenticated:
        return render(request, "firstAid.html")
    else:
        return redirect("landing")

def doctorRecommendation(request):
    if request.user.is_authenticated:
        return render(request, "doctor.html")
    else:
        return redirect("landing")

def docMaps(request):
    if request.user.is_authenticated:
        return render(request, "maps.html")
    else:
        return redirect("landing")
