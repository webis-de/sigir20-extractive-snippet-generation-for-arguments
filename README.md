# ArgsRank

sudo docker build . -t args_snippet_gen
curl -H "Content-Type: application/json" --data '{"arguments":[{"id": 123, "text": "text"}]}' http://127.0.0.1:5000/snippets
