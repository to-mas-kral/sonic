#!/usr/bin/env fish

function compare_images
    set reference_file $argv[1]
    set test_base $argv[2]
    set output_csv $argv[3]

    touch "$output_csv"

    if test -z "$reference_file" -o -z "$test_base"
        echo "Usage: compare_images <reference_file> <test_base>"
        exit 1
    end

    echo "Samples,MSE","FLIP" > $output_csv
    printf "%-30s | %-20s | %-10s\n" "Samples" "MSE" "FLIP"
    printf "%-30s | %-20s | %-10s\n" "------------------------------" "-------------------" "----------"

    set reference_base (basename "$reference_file" | cut -d. -f1)
    set reference_ext (basename "$reference_file" | sed "s|$reference_base\.||")

    for test_image in $test_base*
        if test -f "$test_image"
            set mse (compare -metric MSE "$reference_file" "$test_image" null: 2>&1)
            set mse (string split " " $mse)[1]

            set test_filename (basename "$test_image")

            set num_samples_string (string match -r '.*-(\d+)\.' $test_filename | string replace -r '.*-(\d+)\.' '$1')
            set num_samples (string split " " $num_samples_string)[2]
            set num_samples (string split " " $num_samples_string)[2]

            if test -z "$num_samples"
                continue
            end

            set flip (flip -v 1 -nexm -nerm -r $reference_file -t $test_image)
            set flip_mean_string (string match -r 'Mean: ([0-9.]+)' $flip)
            set flip_mean (string split " " $flip_mean_string)[2]

            printf "%-30s | %-20s | %-10s\n" "$num_samples" "$mse" "$flip_mean"
            echo "$num_samples,$mse","$flip_mean" >> $output_csv
        end
    end

    echo "Results saved to $output_csv"
end

# Call the function with provided arguments
compare_images $argv[1] $argv[2] $argv[3]
