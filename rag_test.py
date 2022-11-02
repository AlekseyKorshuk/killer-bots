from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import RAGenerator, DensePassageRetriever

# document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

# Initialize DPR Retriever to encode documents, encode question and query documents
retriever = DensePassageRetriever(
    document_store=None,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True,
    embed_title=True,
)

from haystack.nodes import Seq2SeqGenerator


generator = Seq2SeqGenerator(model_name_or_path="facebook/bart-large-cnn")

docs = [
    """![Unosquare Blog](https://blog.unosquare.com/hubfs/Claudia%20Blog%20Post.001.jpeg)

In my first years as a front-end developer, I was part of a team of developers where the Tech Leads were very worried about "cleaning" the code.

In retrospect, I understand what they were asking of me: that my code needed to be very descriptive so that anyone could understand it. If someone reviewed my code, they should realize quickly and easily how we solved the user stories of the client. The end result would be readable and maintainable code.

And how could we achieve that? Well, it's not all that complicated and the Tech Leads were right, we needed clean code. Since then I have had mentors that helped me a lot, and they helped uncover a desire for continuous learning that continues to this day using certain [courses and books](https://www.oreilly.com/learning-paths/learning-path-clean/8204091500000000001/). And, the concept of writing clean code has stuck with me.

In this post, I focus on some points that we need to have in mind when we write clean code, focusing on JavaScript.

1\. Use descriptive names

Make the names of the variables and the names of the functions must be as descriptive as they can be. For example, suppose you want to make some references to the character and last name. In that case, you might use a variable that calls "namePhysician" and "lastNamePhysician" instead of using "name." Even worse if you use "n." Why? Because with that syntax, anybody can understand what value the variable contains.

And what about a function? Well, the name of a function must start with a verb. For example, if I have a function that returns the physician's name, I can create a function that calls "getPhysicianName" instead of using a function that only calls "name."

![](https://blog.unosquare.com/hs-fs/hubfs/image-png-Oct-26-2021-12-27-56-76-PM.png?width=807&name=image-png-Oct-26-2021-12-27-56-76-PM.png)

2\. Use empty lines to create a readable code

With empty lines, we can add legibility to our code. An extra line will be beneficial to identify in an easy way where the functions end. Also, we can use empty lines to separate the variable's declaration from the function's operation. Finally, we might add an extra line before the returned value if you want.

![](https://blog.unosquare.com/hs-fs/hubfs/image-png-Oct-26-2021-12-29-29-01-PM.png?width=807&name=image-png-Oct-26-2021-12-29-29-01-PM.png)

3\. Do not send more than three parameters into a function

Remember, we must make a readable function. It is easier to have three parameters and follow their logic inside the function than to have a bunch of parameters and try to find where the parameters are used.

If we need to use more than three parameters, we can send one object to the function and use the keys of the object as we need. Review the following point if you need to send many parameters into a function.

![](https://blog.unosquare.com/hs-fs/hubfs/image-png-Oct-26-2021-12-29-59-68-PM.png?width=807&name=image-png-Oct-26-2021-12-29-59-68-PM.png)

4\. Remember the functions must do only one thing

I know that sometimes we are in a hurry or want to solve our problem with one function, and we write a function that does many things. Avoid doing that. A function must do one thing. It is better to have two or more short and easy functions than to have one complex function.

![](https://blog.unosquare.com/hs-fs/hubfs/image-png-Oct-26-2021-12-30-56-69-PM.png?width=807&name=image-png-Oct-26-2021-12-30-56-69-PM.png)

5\. Functions must be small

If you need to do a function with many lines, you must consider that maybe it is more accurate to use a class instead of a function. Remember your function must do only one thing.

6\. Reduce the number of characters in a line

Keep in mind that we want to create a code that is easy to read. Avoid having long lines of code. The size of a line must fit in your screen so that you don't need to do a horizontal scroll. Remember, there are a lot of tools like prettier that allow you to control the format of the code.

7\. Avoid using comments

Sometimes it is difficult to maintain the code. Imagine if we have to maintain the comments too! How can we avoid using a comment? We can use descriptive names. If our code is understandable, we don't need a comment. If we create a comment and somebody has to change the code, we cannot confirm that this person updated the comment. Remember that if you are using a comment, it is because our code is not clear enough. But maybe you think it is sometimes necessary to add a comment, if that's the case, use comments but only in some exceptional cases.

8\. Create a descriptive message when you create a commit

When we create a commit, we have to write a descriptive message. That message could be helpful if we want to remember what our code is doing some months later. Avoid messages that do not give us much information. If you only write messages like "refactoring," perhaps this could not be clear enough for the following developers.

![](https://blog.unosquare.com/hs-fs/hubfs/image-png-Oct-26-2021-12-31-31-58-PM.png?width=807&name=image-png-Oct-26-2021-12-31-31-58-PM.png)

9\. Use Unit Test and practice Test Driven Development

I know that sometimes we think that unit tests are a waste of time. But believe me, it is false. Indeed, unit tests do work. Imagine that somebody is working with the code you created some months ago, and the new developer creates a code that is solving a thing, but it is breaking other functionality. How can you or the other developer know that the code is breaking? Well, if the project has unit tests, it could be easy to identify those problems. Yes, I know that sometimes we have to deliver many things in a very short time, but this extra time that you invest in unit tests can save a lot of time in the future.

As a good practice, first, create your unit test; obviously, it will fail, continue creating your routine or update the code, and finally rerun your unit test; this will run successfully. The advantage of working in that way is that you can have a better approach to solving the problem you are facing.

10\. Learn Design Patterns

"Design Patterns" is a very broad topic. As an introduction, if you study Design Patterns, you can know the solutions that some master developers had found when facing some common problems in their software. And it lets us avoid reinventing the wheel.

So I highly recommend reading the book: [Design Patterns, Elements of Reusable Object-Oriented Software](https://www.amazon.com/dp/0201633612/ref=cm_sw_em_r_mt_dp_4RBFSWVPGDWE741SCSH0), written by Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides.

A brief story

Finally, I want to share with you an experience that I had with a project that did not use clean code. Calling back to my earlier anecdote, when I first started as a developer the team was focused on clean code. At that time, I didn't understand the importance.

I only saw the value of clean code some months later when I participated in a new project. The objective of this project was to migrate a legacy product to a modern environment. It started with a team, but some of the members were changed after some time. The code that I began working with, well, let me just say that it wasn't exactly "clean." Most of what I learned, and speak about in this article, came from the time consuming efforts of this new project. It was difficult to write the code we wanted, and especially challenging when trying to fix the existing code. 

Right now, I don't have news about that project, but they are still having some problems to this very day. Three years down the line, the project continues to be extended and I'm not confident that they will ever have a stable app. The company is still stuck to legacy products. If they would code following the good practice of clean code, they could avoid those problems.

Keep in mind that as a developer, we should not only be worried about delivering things. It is more than that; it is about thinking about the next developer that should support the code. So please, leave your code with the legibility that you want to find in your next project.

"All change is hard at first, messy in the middle and so gorgeous at the end," - Robin Sharma.

**Ready to choose a software outsourcing company?** 

Unosquare is ready to help. We focus on finding the best talent globally and putting together a delivery management practice that is your eyes and ears within our organization. Our distributed agile software development staffing solutions provide the best expertise for your teams in a fast, transparent, and efficient way. To find out more about outsourcing software projects with Unosquare, check out our [blog](https://blog.unosquare.com/)."""
]

result = generator.predict(
    query='how to write clean code?',
    documents=[Document(doc) for doc in docs],
    top_k=1
)

print(result)