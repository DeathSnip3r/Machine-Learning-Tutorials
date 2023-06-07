#!/bin/bash

n=50  # Specify the number of times you want the script to run
count=0  # Initialize a counter

while [ $count -lt $n ]; do
    python cnn.py
    echo "Python script finished. Restarting..."
    sleep 1
    ((count++))  # Increment the counter
done

# After the loop has run for 'n' times
echo "Script ran for $n times. Executing another script..."
# Add your command to execute the other script here
python runAllModels.py
