regex="[0-9]*:[0-9]* python3 main.py $1 $2"
echo $regex
processes=`ps aux | grep -e "$regex" | awk '{print $2}'`

for process in $processes
do
    echo Killing process $process
    sudo kill -9 $process
done
