for file in ./*
do
	  ffmpeg -i "$file" -filter_complex "color=black,format=rgb24[c];[c][0]scale2ref[c][i];[c][i]overlay=format=auto:shortest=1,setsar=1" tmp.png
	  mv tmp.png "$file"
done
