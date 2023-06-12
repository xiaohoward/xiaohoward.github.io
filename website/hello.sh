#!/bin/bash

# Function to calculate the center position
get_center_position() {
    local term_width=$(tput cols)
    local text_length=${#1}
    local padding=$(( (term_width - text_length) / 2 ))
    echo $padding
}

# Clear the screen
clear

# Type four empty lines
for ((i=0; i<4; i++)); do
    echo
done

# Simulate typing effect at the center
text="Hello World! This is Howard Xiao."
padding=$(get_center_position "$text")



# Simulate typing effect
delay=0.1  # Adjust this value to control the typing speed

for ((i=0; i<${#text}; i++)); do
    echo -n "${text:$i:1}"
    sleep $delay
done

echo  # Print a newline after the typing effect

