#git checkout --orphan latest
#git add -A
#git add commit_deleting_history.sh
#git commit -am "json forma data for HBP Feature Extraction GUI"
#git branch -D master
#git branch -m master
#git push -f origin master


rm -rf .git
git init
git add .
git commit -m "Initial commit"
git remote add origin https://llbologna@bitbucket.org/llbologna/eg_json_data.git
git push -u --force origin master
