output_filename_arr = ["sbt",'code','ids','nl','nl.char']


lines_to_keep = 100000
'''
for out in output_filename_arr:
    input_filename = "data2/train/"+out
    with open(input_filename, "r") as input_file, open(out, "w") as output_file:
        for line_number, line in enumerate(input_file):
            if line_number < lines_to_keep:
                output_file.write(line)
            else:
                break
'''
with open("data3/train3.jsonl", "r") as input_file, open("data3/train4.jsonl", "w") as output_file:
    for line_number, line in enumerate(input_file):
        if line_number < lines_to_keep:
            output_file.write(line)
        else:
            break
