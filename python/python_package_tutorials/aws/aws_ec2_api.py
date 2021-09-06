"""AWS API script for setting up all of our EC2 machines.  Our class object
AWSHandler() also makes calls to other relevant scripts in this repository. """

import time
import os
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import boto3
import smtplib
import pandas as pd
import argparse

"""
Reference for this project:
https://stackabuse.com/automating-aws-ec2-management-with-python-and-boto3/

Assumptions about the current working directory:
1. There is a file named ec2-keypair.pem.
2. There is folder named template which has all the jupyter notebooks for the
    class/.
3. There is a file named email_credentials.txt which has the password for the
    bot gmail address.
4. There is a file users.csv where the first column is full names and the
    second is email addresses for students.
5. You are a collaborator on the machine_learning_aws repo and don't need to
    manually provide any credentials to push.
"""


class AWSHandler(object):
    """Main object for starting and stopping instances, creating new
    instances, retrieving instance and user information, and sending emails
    to users for information on their assigned AWS instances.
    """
    def __init__(self, path, read=False):
        if not read:
            full_path = os.path.join(os.getcwd(), path)
            users = pd.read_csv(full_path, header=0)
            if any(['@' in field for field in list(users)]):
                users = pd.read_csv(full_path, header=None)
                users.columns = ['name', 'email']
            usernames = users['email'].apply(lambda email: re.sub('[^0-9a-zA-Z]+', '', email.split('@')[0]).lower())
            users['username'] = usernames
            print(users.head(3))
            self.users = users
            self.users.to_csv('handler_state.csv')
        else:
            self.users = pd.read_csv(os.path.join(os.getcwd(), 'handler_state.csv'))

    def generate_keypair(self):
        """Given that you have properly set up the AWS CLI, generate a .pem file.
        This cannot be done if someone else has already"""
        ec2 = boto3.resource('ec2')
        key_pair = ec2.create_key_pair(KeyName='julian3')
        KeyPairOut = str(key_pair.key_material)
        with open('ec2-keypair.pem', 'w+') as f:
            f.write(KeyPairOut)
        print("Keypair written")

    def get_instance_info(self):
        """Retrieve instance-related information using
        instances that are CURRENTLY RUNNING and associated with the .pem key
        file in this directory.

        Returns:
            A list of [instance_id, instance_type, ip_address, current_state]
            lists indexed by each instance.
        """

        client = boto3.client('ec2', region_name="us-east-1")
        data = client.describe_instances()

        instance_info = list()
        for reservation in data['Reservations']:
            for instance_dict in reservation['Instances']:
                uid = instance_dict['InstanceId']
                instance_type = instance_dict['InstanceType']
                state = instance_dict['State']['Name']
                if 'PublicIpAddress' in instance_dict:
                    ip_address = instance_dict['PublicIpAddress']
                else:
                    ip_address = None
                if state != 'running':
                    continue
                instance_info.append([uid, instance_type, ip_address, state])
        return instance_info

    def get_instances(self):
        """Get all instance objects."""

        ec2 = boto3.resource('ec2')
        instances = list()
        for instance_id, _, _, _ in self.get_instance_info():
            instances.append(ec2.Instance(instance_id))
        return instances

    # wait for all the instances to be in a desired set of states.
    # Retry every 10 seconds.
    def wait_for_instances(self, target_states=['running']):
        for target_state in target_states:
            assert target_state in ['pending', 'running', 'stopping', 'stopped',
                                    'shutting-down', 'terminated']
        instances = self.get_instances()
        states = [instance.state['Name'] for instance in instances]
        ready = all([state in target_states for state in states])
        while not ready:
            print(
                'Waiting for instances to reach %s states. Current states: '
                '%s' % (
                    target_states, sorted(states)))
            time.sleep(10)
            instances = self.get_instances()
            states = [instance.state['Name'] for instance in instances]
            ready = all([state in target_states for state in states])
        print('Instances are ready!')

    def terminate_instances(self):
        """Terminate all EC2 instances.
        """

        # Get instances and iteratively terminate them
        instances = self.get_instances()
        for instance in instances:
            try:
                instance.terminate()
            except:
                print("Unable to terminate instance %s" % instance)

        # After creating instances, hang
        self.wait_for_instances(['terminated'])

    def hibernate_instances(self):
        """Stop all instances, but don't terminate.

        NOTE: Running this class method will ensure that the owner of these
        instances will not incur EC2 usage charges, but it will incur EBS
        storage costs.
        """

        # Get instances and iteratively stop them
        instances = self.get_instances()
        for instance in instances:
            try:
                instance.stop()
            except:
                print("Unable to hibernate instance %s" % instance)

        # Hang until the instances are ready.
        self.wait_for_instances(['stopping', 'stopped', 'terminated'])

    def mail_to_list(self):
        """Send an email to everyone in the provided csv."""

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login('machinelearning.uruguay@gmail.com', "support_vector_machine")

        num_users = self.users.shape[0]
        for idx, (user, email, username, ip_address, instance_id) in self.users.iterrows():
            if not isinstance(user, str):
                print('I encountered an invalid user! Skipping.')
                continue
            body = """\
               Hola %s,


               Below is your login information for today.
               Today's password: pantalones

               Windows users: please be sure to restart your computer every day before class.
               Mac users and users running Linux: Please
               copy and paste the following commands into
               your command line.
               Windows users: paste the following commands
               into Git Bash, and please remember to RESTART or SHUT DOWN
               your computer before coming to class.

               1. Set up ssh port forwarding:
               ssh -o "StrictHostKeyChecking no" -NfL 5005:localhost:8888 ubuntu@%s
               Then type the password.

               If you get an error that says something like:
               bind [127.0.0.1]:5005: Address already in use

               Mac and Linux users, run this:
               pkill -f 5005
               Then retry step 1

               2. Go to your web browser (such
               as Chrome) and type:
               localhost:5005

               This will take you to your AWS Jupyter
               notebooks! If this worked for you, you're all set!


               Mucho amor,
               GSL Uruguay Technical Team
               """ % (user, ip_address)

            msg = MIMEMultipart()
            msg['From'] = 'machinelearning.uruguay@gmail.com'
            msg['To'] = email
            msg['Subject'] = "Updated Login Instructions"
            msg.attach(MIMEText(body, 'plain'))
            text = msg.as_string()
            server.sendmail('machinelearning.uruguay@gmail.com', email, text)
            print('Sent email %s out of %s' % ((idx + 1), num_users))

        # Notify user that emails have finished sending
        print('Done sending emails.')

    def map_machine_info(self):
        """Update the ip_address and instance_id fields in the self.users dataframe; 1:1 machine to user mapping."""

        info_df = pd.DataFrame([(ip_address, instance_id) for instance_id, _, ip_address, _ in self.get_instance_info()])
        info_df.columns = ['ip_address', 'instance_id']
        self.users = pd.concat([self.users, info_df], axis=1)
        self.users.to_csv('handler_state.csv')

    def intelligent_overwrite(self):
        """Helper function for backing up the machines - can be configured so as
        to ensure that user's content is only overwritten in an intelligent
        way."""

        # Notify user that we are using custom overwrite
        print("Making custom overwrites")

        root_path = os.path.join(os.getcwd(), "student_code")
        users = os.listdir(root_path)

        # Iterate through users
        for user in users:
            daily_user = os.path.join(root_path, user, "daily_user")
            try:
                files = os.listdir(daily_user)
            except FileNotFoundError:
                print("Daily User path not found")
                files = []
            # Iterate through files
            for file in files:
                print(os.path.isdir(os.path.join(daily_user, file)))
                if os.path.isdir(os.path.join(daily_user, file)) and \
                        not file.startswith("."):
                    # Change dir, copy contents, change dir
                    os.chdir(daily_user)
                    cp_command = "cp -r %s ../" % (file)
                    os.system(cp_command)
                    os.chdir(root_path)
            # Remove daily user
            remove_command = "rm -r %s" % (daily_user)
            os.system(remove_command)

        # Notify that custom overwrite has been finished
        print("Overwrite finished!")

    def backup_machines(self, custom_overwrite=False):
        """Back up student-populated content from the
        course to GitHub in the 'daily_user' sub-directory.
        """

        assigned_machines = self.users[self.users['name'].notnull()]

        # save the dirs to machine_learning_aws/studient_copies
        root_save_dir = os.path.join(os.getcwd(), 'student_code')

        # Pem file
        credential_path = os.path.join(os.getcwd(), 'ec2-keypair.pem')

        for _, row in assigned_machines.iterrows():
            print(row)
            idx, name, email, username, ip_address, instance_id = row
            local_save_dir = os.path.join(root_save_dir, username)
            print("Local save dir %s " % (local_save_dir))
            print("ID address % s" % (ip_address))
            scp_command = 'scp -r -i %s -o "StrictHostKeyChecking no" -r ' \
                          'ubuntu@%s:/home/ubuntu/machine_learning_aws' \
                          '/daily_user/  %s' % (
                              credential_path, ip_address, local_save_dir)
            os.system(scp_command)

        # Overwrite in an intelligent way (i.e. only files from that day)
        # NOTE: This is currently hardcoded
        if custom_overwrite:
            self.intelligent_overwrite()

        # Now push changes to GitHub
        os.system('git add .')
        os.system('git commit -m "Backed up Jupyter Notebooks %s"' % (
            time.time()))
        os.system('git push')

        # Notify user that machines are backed up
        print("Machines have been backed up!")


    def start_instances(self, count, ami, instance_type='t3a.xlarge'):
        """Spin up n new EC2 instances from scratch, then hang until they're 'running'"""

        if ami == 'ubuntu':
            ami = 'ami-00a208c7cdba991ea'
        ec2 = boto3.resource('ec2', region_name="us-east-1")
        print('Starting %s instances of type %s and AMI %s.' % (count, instance_type, ami))
        # Now create instances using AMI
        ec2.create_instances(
            ImageId=ami,
            MinCount=count,
            MaxCount=count,
            InstanceType=instance_type,
            KeyName='ec2-keypair'
        )
        # Hang until the instances are ready
        self.wait_for_instances(['running', 'terminated', 'shutting-down'])
        print('Sleeping before mapping machine information')
        time.sleep(60)
        self.map_machine_info()

    def start(self, ami, instance_type):
        self.start_instances(count=self.users.shape[0], instance_type=instance_type, ami=ami)
        # self.mail_to_list()
        print('Done starting up.')


if __name__ == "__main__":
    handler = AWSHandler("users.csv", read=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--backup', action='store_true')
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--info', action='store_true')
    parser.add_argument('--path', default='users.csv')
    parser.add_argument('--ami', default='ami-0c505f47bff8ab1b2')
    parser.add_argument('--type', default='t3a.2xlarge')  # g3.4xlarge for GPUs
    parser.add_argument('--email', action='store_true')
    args = parser.parse_args()

    assert sum([int(args.start), int(args.stop),
                int(args.backup), int(args.info), int(args.email)]) == 1, \
        'Must select exactly 1 among: start, stop, backup, email, and info'

    if args.info:
        handler.map_machine_info()

    if not args.start:
        handler = AWSHandler(args.path, read=True)
    else:
        handler = AWSHandler(args.path, read=False)

    if args.email:
        handler.mail_to_list()

    if args.stop:
        assert input('Are you sure you want to kill all the machines? :( ') == 'YES'
        handler.terminate_instances()
    elif args.start:
        handler.start(ami=args.ami, instance_type=args.type)
    else:
        handler.backup_machines()

