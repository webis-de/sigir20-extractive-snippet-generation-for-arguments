
# Paper: Extractive Snippet Generation for Arguments

This is the code for the paper *Extractive Snippet Generation for Arguments*.

Milad Alshomary, Nick Düsterhus, Henning Wachstmuth


    @InProceedings{elbaff:2020,
      author =              {Milad Alshomary, Nick Düsterhus, Henning Wachstmuth},
      booktitle =           {The 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
      month =               July,
      publisher =           {SIGIR},
      site =                {Xi'an, China},
      title =               {{Extractive Snippet Generation for Arguments}},
      year =                2020
    }

# ArgsRank

### Running Code on Docker:
    docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . -t args_snippet_gen

    nvidia-docker run -u $(id -u):$(id -g) --name args-snippet-generation -p 5000:5000 -it args_snippet_gen:latest bash

    RUN CUDA_VISIBLE_DEVICES=5 waitress-serve --listen=0.0.0.0:5000  --call 'argsrank:create_app' >info.log 2> error.log


### Sending http request to generate snippets:

 `curl -H "Content-Type: application/json" --data '{"arguments":[{"id":"5", "text":"The Supreme Court decided that states can not outlaw abortion because Prohibiting abortion is a violation of the 14th Amendment, according to the Court, and the constitution.. Outlawing abortion is taking away a human right given to women.. in reality, a fetus is just a bunch of cells.. It has not fully developed any vital organs like lungs.. This means that an abortion is not murder, it is just killing of cells in the wound.. If the child has no organs developed that would be vital for the baby to survive outside the wound, than having an abortion is not murder."},{"id":"1","text":"In 2011 there were about 730,322 abortions reported to the centers for disease control.. There are about 1.7% of abortion of womens ages from 15-44 each year.. Women who already had abortion earlier in there life time have abortion again.. At the age of 45 a women will have at least one abortion.. By the 12th week of pregnancies 88.7% of women have abortion.. In the U.S. black women are 3.3 times likely to have an abortion than white women."}]}' http://127.0.0.1:5000/snippets`
