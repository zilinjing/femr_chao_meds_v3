# code_dict = {"LOINC/26515-7//UCUM/10*9/L":"Platelets [#/volume] in Blood","LOINC/2160-0//UCUM/mg/dL":"Creatinine [Mass/volume] in Serum or Plasma"
#              ,"LOINC/1968-7//UCUM/mg/dL":"Bilirubin.direct [Mass/volume] in Serum or Plasma","LOINC/2708-6//UCUM/%":"Oxygen saturation in Arterial blood"}

'''
Goal: Get the distribution of inter-event time for four codes. Process each patient in parallel and collect inter-event time. Plot each distribution
'''
import meds_reader
import functools
import numpy as np
import os
import matplotlib.pyplot as plt

code_dict = {"LOINC/26515-7//UCUM/10*9/L":"Platelets","LOINC/2160-0//UCUM/mg/dL":"Creatinine"
             ,"LOINC/1968-7//UCUM/mg/dL":"Bilirubin","LOINC/2708-6//UCUM/%":"Spo2"}

LABELS = list(code_dict.values())

def get_interval(subject_iter,code_dict):
    # interval_list = {"Platelets":[],"Creatinine":[],"Bilirubin":[],"Spo2":[]}
    interval_time_list = {lab:[] for lab in list(code_dict.values())}
    for subject in subject_iter:
        # interval_time_list_sub = {lab:[] for lab in list(code_dict.values)}
        last_time = {lab: None for lab in list(code_dict.keys())}
        # print(f"subject is {subject}")
        for event in subject.events:
            # the event belongs to our interested lab items
            if event.code in code_dict and event.time is not None:
                prev_time = last_time[event.code]
                if prev_time is not None:
                    interval_time_list[code_dict[event.code]].append((event.time-prev_time).total_seconds()/86400.0)
                last_time[event.code] = event.time
        # print(interval_time_list)
    return interval_time_list

def aggre(init,chunk):
    for k in init:
        init[k].extend(chunk[k])
    return init


# def plot_distributions_days(all_intervals_days, bins=80, clip_q=0.99):
#     for lab, vals in all_intervals_days.items():
#         x = np.asarray(vals, dtype=float)
#         x = x[np.isfinite(x)]
#         x = x[x >= 0]

#         if x.size == 0:
#             print(f"[warn] No intervals for {lab}")
#             continue

#         xmax = np.quantile(x, clip_q)
#         x_plot = x[x <= xmax]

#         plt.figure()
#         plt.hist(x_plot, bins=bins, density=True)
#         plt.title(f"Inter-event time: {lab}")
#         plt.xlabel("Days")
#         plt.ylabel("Density")
#         plt.tight_layout()
#         plt.show()


def plot_distributions_days(all_intervals_days, out_dir="interval_plots", bins=80, clip_q=0.99):
    os.makedirs(out_dir, exist_ok=True)

    for lab, vals in all_intervals_days.items():
        x = np.asarray(vals, dtype=float)
        x = x[np.isfinite(x)]
        x = x[x >= 0]

        if x.size == 0:
            print(f"[warn] No intervals for {lab}")
            continue

        xmax = np.quantile(x, clip_q)
        x_plot = x[x <= xmax]

        plt.figure()
        plt.hist(x_plot, bins=bins, density=True)
        plt.title(f"Inter-event time: {lab}")
        plt.xlabel("Days")
        plt.ylabel("Density")
        plt.tight_layout()

        fname = os.path.join(out_dir, f"inter_event_{lab}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print("Saved:", fname)

def main():
    meds_path = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader/"
    db = meds_reader.SubjectDatabase(meds_path, num_threads=32)

    init_result = {lab: [] for lab in LABELS}
    worker = functools.partial(get_interval, code_dict=code_dict)

    all_interval_days = functools.reduce(aggre, db.map(worker), init_result)
    print({k: len(v) for k, v in all_interval_days.items()})
    plot_distributions_days(all_interval_days)

    db.terminate()  # important to clean up processes

if __name__ == "__main__":
    main()