# Installation

This installation process is intended for a Linux OS which can also be performed on a Mac. Windows may not be the best
environment and may require too many workarounds.

The goal is to set up a test environment using the vector framework (Jina) while
being able to make network API Curl calls (Postman) to communicate. Once set up there should be a way to program
decisions for the API endpoints in the vector framework to perform any AI tasks you need. This document just describes
how to set up the test environment but does not include any programmed tasks. Those have to be created depending on 
what you want to do.

For starters you will want:
-	A Python IDE. This is to view the code with error hints. I use PyCharm since they have a free community version
    (https://www.jetbrains.com/pycharm/download/)
-	You will need a way to access the API. I use Postman for this because it is easy to use and I have built a
    collection for this test already which is included. (https://www.postman.com/downloads/)
-	A system with a Unix based OS either Linux or Mac with command line access. I use Ubuntu. If your test server
    is not the development server you may need a way to transfer files over FTP. For that I use FileZilla
    (https://filezilla-project.org/)
-	Read the docs for Jina: https://docs.jina.ai/
-	Read the docs for DocArray: https://docarray.jina.ai/ 

## Install Jina 

Install Jina itself.
You must have Python 3.7+ and PyPi installed
(more info: https://docs.jina.ai/)

To install with PyPi run: `pip install -U Jina`

Now you can edit the host and port if needed on the `flow.yml` file

To run it just run the command: `python3 flow.py` from the root directory

After running, you can connect with Postman or any other API service using your
IP address as host on port 1234

for API docs you can see them in a web browser at:
http://{Your Jina Instance IP Addr}:1234/docs
or
http://{Your Jina Instance IP Addr}:1234/redoc

## Set up Postman

Now you can play around with the code. I use Postman to test the API connections. Once you are at this point you can
just import the Postman configuration from the files called `Jina.postman_collection.json`. This will give you the
endpoints for this collection. All you will have to do is change the `url` and `port` variables to your setup.
(https://learning.postman.com/docs/getting-started/importing-and-exporting-data/)

------------------------------------------------------------------------------------------------------------------------

### Where to find the stuff to work on:
Jina is a vector framework. They provide a way to build a flow which can use AI tools like Spacy or TensorFlow to
perform tasks. Each task in a flow is called an Executor. In this installation there are no Executors.

Jina also comes with its built in tool called DocArray that is the way it manages documents. It takes unstructured data
including images, videos, text, etc and standardizes them in a way that makes all of this stuff possible. Within this
project you will be manipulating documents within a document array. These will be vectorized and returned to the user
so we can play with all the data without having to load it all at once. DocArray or Jina
does not store data without a data store.

in the Postman Collection you will find a few endpoints. These are used to test passing data.
-	Index
-	Search
-	Delete
-	Update

These do what they say they do. There are comments describing what the functions should do in the file.

There are two goals here:
1.	Get the project installed and verify but sending a Postman request to any endpoint without errors
2.	Write Python code to make the endpoints do what they are supposed to do.

The first task should be pretty simple since there is a step by step guide in this document but the second task is
more challenging since the only guidance is your imagination and your drive to learn.

### File Structure

Here is a description of the important files for this project:
- `flow.py` is the file that gets executed. This has to run and stay open for Postman to work.
    If you need to make changes to the response that goes between Postman and the Database this is where it all starts.
    If you do make any changes and upload them don’t forget to stop and then restart this file or you
    will not see your updates.
- `flow.yml` is the flow configuration file. This is where we add executors into the flow and configure them
    based on the requirements. To add an executor this is where those changes will be made. It also holds all of the
    environment variables used within the project. You can also change host and port here along with any other
    global variables you need to set.
- `docker-compose.yml` is just the configuration file to create the Weaviate Vector DB in a Docker image.
    No need to mess with this file but just make sure it is running or else the install wont work.
- `exec/` is a directory where all the executors are installed for these projects. Each executor should have its own sub folder
    within this folder to keep everything organized.
    Each executor will have its own folder within the exec/ directory so there can be multiple files if needed. If you
    are importing an Executor directly from Jina HUb then you dont need to make a directory for it.

Another note: If you add an executor to the flow or make modifications you can visualize the flow by running
this command in the root directory: `python3 flow.py svg`.
This will create a svg image called `flow.svg` of the flow in the root directory that you can open up in a web browser.
This can be handy to visualize any changes or updates made to the flow.
