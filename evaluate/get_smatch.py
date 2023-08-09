import smatch
import sys
import csv

def get_precision_recall_f1(output_amr, target_amr):
    result = smatch.get_amr_match(output_amr, target_amr)
    # total_precision = 0.0
    # total_recall = 0.0
    # total_f1 = 0.0

    # ITERATION_COUNT = 2500
    # for _ in range(ITERATION_COUNT):
    #     precision, recall, f1 = smatch.compute_f(*result)
    #     total_precision += precision
    #     total_recall += recall
    #     total_f1 += f1

    # return (
    #     total_precision / ITERATION_COUNT,
    #     total_recall / ITERATION_COUNT,
    #     total_f1 / ITERATION_COUNT
    # )

    precision, recall, f1 = smatch.compute_f(*result)
    assert 0.0 <= precision <= 1.0, f"Precision out of range: {precision}"
    assert 0.0 <= recall <= 1.0, f"Recall out of range: {precision}"
    assert 0.0 <= f1 <= 1.0, f"F-1 out of range: {precision}"

    smatch.match_triple_dict.clear()

    return precision, recall, f1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        output_amr = "( t / tuang :arg0 ( a / adonan ) :arg1 ( a2 / adonan ) :arg1 ( d / dalam :mod ( l / loyang ) ) :time ( s / setelah :mod ( d2 / dingin ) ) )"
        target_amr = "( t / tuang :arg1 ( a / adon ) :location ( d / dalam :mod ( l / loyang ) ) :time ( t2 / telah :mod ( a2 / adon :mod ( d2 / dingin ) ) ) )"
        precision, recall, f1 = get_precision_recall_f1(output_amr, target_amr)
        print(f"P: {precision:.3f}")
        print(f"R: {recall:.3f}")
        print(f"F: {f1:.3f}")
    else:
        csv_rows = [["output", "target", "precision", "recall", "f1"]]
        _, output_amr_path, target_amr_path, output_csv_path = sys.argv
        with open(output_amr_path, mode="r") as f:
            output_amr_lines = f.readlines()
        
        with open(target_amr_path, mode="r") as f:
            target_amr_lines = f.readlines()

        count = 0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        for output_amr, target_amr in zip(output_amr_lines, target_amr_lines):
            output_amr = output_amr.strip()
            target_amr = target_amr.strip()
            precision, recall, f1 = get_precision_recall_f1(output_amr, target_amr)
            print(f"Line {count + 1}:", precision, recall, f1)
            csv_rows.append([output_amr, target_amr, precision, recall, f1])
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1

        global_avg_precision = total_precision / count
        global_avg_recall = total_recall / count
        global_avg_f1 = total_f1 / count

        print(f"P: {global_avg_precision:.3f}")
        print(f"R: {global_avg_recall:.3f}")
        print(f"F: {global_avg_f1:.3f}")

        with open(output_csv_path, mode="w", newline="") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(csv_rows)
