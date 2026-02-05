
import pandas as pd
import os

# File handling inside the function is safer to ensure you always read the latest version
def add_new(user_id, password):
    # Load current data
    if os.path.exists('credentials.csv'):
        df = pd.read_csv('credentials.csv')
    else:
        df = pd.DataFrame(columns=['user_id', 'password'])

    # 1. Correct way to check if user exists
    if df[df['user_id'] == user_id].empty:
        
        if len(str(password)) > 8:
            # 2. Correct way to append a row
            new_row = pd.DataFrame({'user_id': [user_id], 'password': [str(password)]})
            df = pd.concat([df, new_row], ignore_index=True)
            
            # 3. CRITICAL: You must write it back to the file!
            df.to_csv('credentials.csv', index=False)
            
            return {'message': "Successfully added user id to database", 'status': True}
        else:
            return {'message': "Password must be more than 8 characters", 'status': False}
    else:
        return {'message': "User ID already exists", 'status': False}
    
def validate_user(user_id, password):
    df = pd.read_csv('credentials.csv')
    df['user_id'] = df['user_id'].astype(str)
    df['password'] = df['password'].astype(str)

    if df[df['user_id']==user_id].empty:
        return {'message':'user does not exist', 'status':False}
    else:

        l_password = df[df['user_id']==user_id].iloc[0]['password']
        print(l_password)
        print(password)
        row = (l_password.strip() == str(password).strip())
        if row:
            return {'message':'login successfully','status':True}
        else:
            return {'message':'check your password', 'status':False}
        
if __name__ == '__main__':
    print(validate_user('user1', 123456789))