#!/usr/bin/env fish

# Check if the required arguments are provided
if test (count $argv) -lt 4
    echo "Usage: crop_exr_images.fish <file_path> <x> <y> <width> <height>"
    exit 1
end

# Extract the file path and crop parameters
set file_path $argv[1]
set x $argv[2]
set y $argv[3]
set width $argv[4]
set height $argv[5]
set outfolder $argv[6]

# Ensure the file exists
if not test -f $file_path
    echo "Error: File '$file_path' does not exist."
    exit 1
end

# Extract directory and base name
set dir (dirname -- "$file_path")
set filename (basename -- "$file_path")
set base_name (string match -r '^(.*)\.exr$' $filename | tail -n 1)

# Ensure base_name extraction succeeded
if test -z "$base_name"
    echo "Error: File does not have a .exr extension"
    exit 1
end

# Find all matching files in the directory and crop them
for file in $dir/$base_name*.exr
    if test -f "$file"
        set output_file (string replace -a .exr ".png" $file)
        set output_file (string replace .png "$x-$y" $output_file)

        if string match -q '*.png' $output_file
            set output_basefile (basename -- "$output_file")
            set output_path "$outfolder/$output_basefile"

            magick "$file" -gamma 2.2 -crop "$width"x"$height"+"$x"+"$y" "$output_path"
        end
    end
end
