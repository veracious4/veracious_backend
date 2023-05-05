import pandas as pd
import random
import json
from utils import validate_fact, get_verdict_from_trust_score

class TestModel:
    test_dataset = ""
    checkpoint_file_location = ""

    def __init__(self, fileLocation:str):
        '''
            Set the checkpoint file location for further use
            @param fileLocation: Location of the checkpoint file
        '''
        self.checkpoint_file_location = fileLocation


    def __read_json_file_data(self, fileLocation:str):
        '''
        Read data from a Json file
        @param fileLocation: Location of the Json file
        '''
        with open(fileLocation) as file:
            data = json.load(file)
        return data
    
    
    def __write_json_data_to_file(self, fileLocation:str, newData:json):
        '''
        Writes data to a Json file
        @param fileLocation: Location of the checkpoint file
        @param newData: new Json data that is to be overwritten on the existing file
        '''
        with open(fileLocation, 'w') as file:
            json.dump(newData, file, indent=2)
    
    def __read_dataset(self):
        '''
            Reads the dataset form a given checkpoint file location
            @param fileLocation: Location of checkpoint file being used
        '''
        # ---------- Read the checkpoint Json file ----------
        checkpoint_data = self.__read_json_file_data(self.checkpoint_file_location)
        data_file_location = checkpoint_data['data_file_location']
        random_state = checkpoint_data["random_state"]
        test_size = checkpoint_data["total_size"]
        
        # ---------- Read and clean the dataset ----------
        dataset = pd.read_json(data_file_location, lines=True)
        dataset.dropna(inplace=True)
        # Drop the unnecessary columns
        dataset.drop(['id', 'evidence', 'annotator_operations', 'challenge'], axis=1, inplace=True)
        # Remove rows containing empty label
        dataset['label'] = dataset['label'].str.strip()
        dataset = dataset[dataset['label'] != '']

        # ---------- Create a sample of test dataset ----------
        test_dataset = dataset.sample(n=test_size, random_state=random_state)
        test_dataset.reset_index(inplace=True, drop=True)
        self.test_dataset = test_dataset
    
    def reset_json_file_to_default(self,
                               dataFileLocation="../data/feverous/feverous_train_challenges.jsonl",
                               testSize=100, 
                               randomState=128):
        '''
        Reset Json checkpoint file to Default parameters.
        @param dataFileLocation: Location of the testing dataset. Default is set to feverous_train_challenges.json file
        @param testSize: Number of enteries to be picked from the testing dataset. Default is set to 100.
        @param randomState: Defines randomState for choosing enteries from the testing dataset. Default is set to 128.
        '''
        initial_checkpoint_data = {"data_file_location": dataFileLocation, 
                                   "random_state":randomState, "total_size": testSize, 
                                   "last_executed":0, "current_accuracy":0}
        self.__write_json_data_to_file(self.checkpoint_file_location, initial_checkpoint_data)
    


    def __perform_automated_testing(self):

        '''
         A helper function that performs make actual calls to the functions to perform testing. This function 
         makes recursive calls to automatically perform testing of the next enteries in the test dataset.
        '''
        checkpoint_data = self.__read_json_file_data(self.checkpoint_file_location)
        
        # ---------- If testing has already beeen completed ----------
        if checkpoint_data["total_size"] == checkpoint_data["last_executed"]:
            print("Your testing has been completed. Final accuracy score your model is: " +
                  str(checkpoint_data["current_accuracy"]))
            print("For more details, please check ", self.checkpoint_file_location)
            print("To perform the testing again, with modified parameters, try resetting the checkpoint file")
            return
        
        # ---------- Some enteries are still to be tested ----------
        last_executed_count = checkpoint_data["last_executed"]
        previous_accuracy = checkpoint_data["current_accuracy"]

        # ---------- Select the next index from the test_dataset ----------
        current_index = last_executed_count
        current_entry = self.test_dataset.iloc[current_index]
        current_claim = current_entry['claim']
        current_label = current_entry['label']

        # ---------- Fact check the new entry ----------
        fact_validation_result = validate_fact(current_claim)
        # fact_validation_result = {"trust_score": random.random()}
        claim_verdict = get_verdict_from_trust_score((float)(fact_validation_result))
        # claim_verdict = get_verdict_from_trust_score((float)(fact_validation_result["trust_score"]))
        claim_accuracy_score = 1 if claim_verdict == current_label else 0
        current_accuracy = (previous_accuracy* last_executed_count + claim_accuracy_score)/(last_executed_count + 1)
        
        # ---------- Update the checkpoint Json file ----------
        checkpoint_data["last_executed"] = last_executed_count + 1
        checkpoint_data["current_accuracy"] = current_accuracy

        self.__write_json_data_to_file(self.checkpoint_file_location, checkpoint_data)
        print("Current accuracy score of your model with " + (str)(last_executed_count+1) + 
              " enteries is: " + (str)(checkpoint_data["current_accuracy"]))

        # ---------- Make a recursive call for next enteries ----------
        self.__perform_automated_testing()

    
    def continue_testing(self):
        '''Resumes Testing from the last entry in the checkpoint file'''

        self.__read_dataset()
        self.__perform_automated_testing()
    

    def start_testing(self):
        '''Starts testing from the beginning of the test dataset, using the default parameters'''
        # self.reset_json_file_to_default()
        self.continue_testing()

def run():    
    try:
        tester = TestModel(fileLocation="./testing/checkpoint.json")
        tester.start_testing()
    except Exception as e:
        print("Something went wrong")
        print(e)

run()
