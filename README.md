# DamenChat
This repository is built to deploy a streamlit app locally and virtually, it is designed to only read .json files that contain markdown content.

This folder contains a requirements.txt which contains the libraries needed to run the code in any environment. To install the dependencies run the command: pip install -r requirements.txt

chroma_db: serves as a folder where the app will save the vectordatabases for each conversation taken -> note: this might lead to some type of error in the future, as we might need to delete the created ones on the server running virtually (so far there have been no issues).

.streamlit: this is an empty file, however it is essential for the deployment of the app, this folder is used by their server to create our equivalent of the secrets.toml file, this folder is where all the API's will be saved and all necessary passwords, never remove this folder.

app.py: Main file that is executed to run the complete app virtually and locally.


First working online version of the app : 09-04-2025

## TODO:
- Maybe restructure the format of the code as everything is run on the same file for now, so for example: one file to design the web and another file that runs the backend of the application.
- Migrate the app within Azure

## IMPORTANT NOTES:
- If you run the app locally some folder will be added into the chroma_db folder, do not push these files into the github.
- You might get an error 529 which means that the servers of Anthropic are overloaded, this does not have to do with the code
- Another error you might get is that you do not have enough credits, this erros basically means we ran out of money and we need to put more money to use the API: https://console.anthropic.com/settings/billing
- Keep in mind the longer the conversations the more we will pay.


