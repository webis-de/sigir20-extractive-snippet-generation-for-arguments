# ArgsRank

### Building docker image:
sudo docker build . -t args_snippet_gen

docker run -u $(id -u):$(id -g) --name args_rank -it -v /home/miladalshomary/Development/argument-snippet-generation/argsrank:/usr/local/argsrank-new args_snippet_gen:latest

### Sending http request to generate snippets:

curl -H "Content-Type: application/json" --data '{"arguments":[{"id":"0191","text":"Abortion is wrong! Abortion Is gross! Abortion is MURDER!!!!"},{"id":"02391","text":"This is a test. Lets look if this works."}]}' http://127.0.0.1:5000/snippets