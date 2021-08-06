#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code demonstrates how to use dedupe with a comma separated values
(CSV) file. All operations are performed in memory, so will run very
quickly on datasets up to ~10,000 rows.

We start with a CSV file containing our messy data. In this example,
it is listings of early childhood education centers in Chicago
compiled from several different sources.

The output will be a CSV with our clustered results.

For larger datasets, see our [mysql_example](http://open-city.github.com/dedupe/doc/mysql_example.html)
"""

import os
import csv
import dedupe
import pandas as pd


def read_data(filename):
    """
    Read in our data from a CSV file to Pandas Dataframe
    """
    print("reading csv data")
    df = pd.read_csv(filename, sep=",", usecols=["Site name","Address", "Zip","Phone"])
    # clean strings
    for col in df.loc[:, df.dtypes == 'object']:
        df[col] = df[col].str.replace("  +","")
        df[col] = df[col].str.replace("\n", "")
        df[col] = df[col].str.strip().str.strip('""').str.strip("'").str.lower().str.strip()
    # clean integers
    for col in df.loc[:, df.dtypes == 'float']:
        df[col] = df[col].astype("Int64").astype(str)
        df.loc[df[col].str.startswith("<NA>"), col] = None
    return df


if __name__ == "__main__":

    ###### Setup
    input_file = "csv_example_messy_input.csv"
    output_file = "csv_example_output.csv"
    settings_file = "csv_example_learned_settings"
    training_file = "csv_example_training.json"

    df_d = read_data(input_file)

    ####### Training

    if os.path.exists(settings_file):
        print("reading from", settings_file)
        with open(settings_file, "rb") as f:
            deduper = dedupe.StaticDedupe(f)

    else:
        # Define the fields dedupe will pay attention to
        #
        # Notice how we are telling dedupe to use a custom field comparator
        # for the 'Zip' field.
        fields = [
            {"field": "Site name", "type": "String"},
            {"field": "Address", "type": "String"},
            {"field": "Zip", "type": "Exact", "has missing": True},
            {"field": "Phone", "type": "String", "has missing": True},
        ]

        # Create a new deduper object and pass our data model to it.
        deduper = dedupe.Dedupe(fields)

        # To train dedupe, we feed it a sample of records. If training file exists, use it
        if os.path.exists(training_file):
            deduper.prepare_training(df_d.T.to_dict(), training_file=training_file, sample_size=15000)
        else:
            deduper.prepare_training(df_d.T.to_dict(), sample_size=15000)

        # ## Active learning
        # Dedupe will find the next pair of records
        # it is least certain about and ask you to label them as duplicates
        # or not.
        # use 'y', 'n' and 'u' keys to flag duplicates
        # press 'f' when you are finished
        print("starting active labeling...")

        dedupe.console_label(deduper)

        deduper.train()

        # When finished, save our training away to disk
        with open(training_file, "w") as tf:
            deduper.write_training(tf)

        # Save our weights and predicates to disk.  If the settings file
        # exists, we will skip all the training and learning next time we run
        # this file.
        with open(settings_file, "wb") as sf:
            deduper.write_settings(sf)

    # ## Clustering, returning pairs that dedupe believes are the same entity

    print("clustering...")

    clustered_dupes = deduper.partition(df_d.T.to_dict(), threshold=0.7)

    print("# duplicate sets", len(clustered_dupes))

    # ## Writing Results

    # Write our original data back out to a CSV with a new column called
    # 'Cluster ID' which indicates which records refer to each other.

    cluster_membership = {}
    cluster_id = 0
    for cluster_id, cluster in enumerate(clustered_dupes):
        id_set, scores = cluster
        # cluster_d = [df_d[c] for c in id_set]
        cluster_d = [df_d.loc[i, :].to_dict() for i in id_set]
        canonical_rep = dedupe.canonicalize(cluster_d)
        for record_id, score in zip(id_set, scores):
            cluster_membership[record_id] = {
                "cluster id": cluster_id,
                "canonical representation": canonical_rep,
                "confidence": score,
            }

    singleton_id = cluster_id + 1

    with open(output_file, "w") as f_output:
        writer = csv.writer(f_output)

        with open(input_file) as f_input:
            reader = csv.reader(f_input)

            heading_row = next(reader)
            heading_row.insert(0, "confidence_score")
            heading_row.insert(0, "Cluster ID")
            canonical_keys = canonical_rep.keys()
            for key in canonical_keys:
                heading_row.append("canonical_" + key)

            writer.writerow(heading_row)

            for row in reader:
                row_id = int(row[0])
                if row_id in cluster_membership:
                    cluster_id = cluster_membership[row_id]["cluster id"]
                    canonical_rep = cluster_membership[row_id]["canonical representation"]
                    row.insert(0, cluster_membership[row_id]["confidence"])
                    row.insert(0, cluster_id)
                    for key in canonical_keys:
                        row.append(canonical_rep[key].encode("utf8"))
                else:
                    row.insert(0, None)
                    row.insert(0, singleton_id)
                    singleton_id += 1
                    for key in canonical_keys:
                        row.append(None)
                writer.writerow(row)

