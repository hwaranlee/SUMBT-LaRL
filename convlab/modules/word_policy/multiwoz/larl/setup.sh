# Download and open pretrained LaRL model
curl -L -o sys_config_log_model.zip "https://drive.google.com/uc?id=1FXyy5WCrAqlhtOg-vorr7UpXjCXj3eGx&export=download"
unzip sys_config_log_model.zip
rm sys_config_log_model.zip

# Extract data used to train LaRL
cd data/
unzip norm-multi-woz.zip