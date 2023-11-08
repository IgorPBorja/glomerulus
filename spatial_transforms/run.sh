code="/path/to/workspace/folder"
inv_imgs="/path/to/save/folder"

img=("" "treino_azan (45).jpg" "treino_he (2).jpg" "treino_picro (13).jpg")

for i in {1..3};
do
	python3 "${code}/features_extract.py" "${inv_imgs}/img${i}/${img[i]}" "${inv_imgs}/img${i}/hist.jpg" "${inv_imgs}/img${i}/normalized.jpg" "${inv_imgs}/img${i}/opponent.jpg" "${inv_imgs}/img${i}/opponent_hist.jpg"
done
