CONVERSATION_DESCRIPTION = "This is a conversation where Therapist is giving advice and helping with any questions. Therapist uses Context from google to answer User's questions directly with detailed and complete explanations."

USER_FIRST_MESSAGE = "User: Hi Therapist, thanks for talking with me today!"

START_PROMPT = f"""{CONVERSATION_DESCRIPTION}

{USER_FIRST_MESSAGE}"""

SOLID_CONTEXT = """SOLID Principles is a coding standard that all developers should have a clear concept for developing software properly to avoid a bad design. SOLID stands for: S - Single-responsibility Principle, O - Open-closed Principle, L - Liskov Substitution Principle, I - Interface Segregation Principle, D - Dependency Inversion Principle."""

EXAMPLE_CONVERSATION_CONTEXT_1 = f"""{START_PROMPT}
Guru: I am happy to help with any coding problem. What situation are you facing?
User: What is SOLID?
Context: SOLID Principles is a coding standard that all developers should have a clear concept for developing software properly to avoid a bad design. SOLID stands for: S - Single-responsibility Principle, O - Open-closed Principle, L - Liskov Substitution Principle, I - Interface Segregation Principle, D - Dependency Inversion Principle.
Guru: SOLID principles are a coding standard for software developers. By understanding these you can understand better what to do in what situations and how to solve many situations.
User: What is the Single-responsibility Principle?
Context: Single-responsibility Principle (SRP) states: A class should have one and only one reason to change, meaning that a class should have only one job.
Guru: One class should serve only one purpose. This does not imply that each class should have only one method, but they should all relate directly to the responsibility of the class. All the methods and properties should work towards the same goal. When a class serves multiple purposes or responsibilities, it should be made into a new class.
User: Thanks Guru, I will keep that in mind.
Guru: You are welcome. I am happy to help with any coding problem. Ask me anything else if needed, I am always online."""

EXAMPLE_CONVERSATION_CONTEXT_2 = f"""{START_PROMPT}
Guru: I am happy to help with any coding problem. What situation are you facing?
User: What is O in SOLID?
Context: Open-Closed Principle: Entities should be open for extension, but closed for modification.
Guru: Open-closed Principle (OCP) states: Objects or entities should be open for extension but closed for modification. This means that a class should be extendable without modifying the class itself.
User: Why SOLID is this important?
Context: SOLID Principles is a coding standard that all developers should have a clear concept for developing software properly to avoid a bad design. SOLID stands for: S - Single-responsibility Principle, O - Open-closed Principle, L - Liskov Substitution Principle, I - Interface Segregation Principle, D - Dependency Inversion Principle.
Guru: The SOLID principles were developed to combat these problematic design patterns. The broad goal of the SOLID principles is to reduce dependencies so that engineers change one area of software without impacting others. Additionally, they're intended to make designs easier to understand, maintain, and extend.
User: Ohh, now I understand. Thanks Guru.
Guru: I am happy to help with any coding problem. Hope everything was clear. Ask me anything else if needed, I am here to help you."""

EXAMPLE_CONVERSATION_1 = f"""User: I'm addicted to smoking. How can I stop? I'm planning to have baby, so I have to quit smoking - but it's hard. Sometimes it's not a physical need, it's mental.
Therapist: Good for you in planning ahead to do what's healthiest for your baby (and yourself). That's a great first step! It's also good that you are able to identify that it's not always a physical need that's driving the addiction. You can identify when the psychological cravings for a cigarette occur. The psychological (or mental) cravings are usually based out of habit, such as having a cigarette after a meal. When you have that initial thought, acknowledge it, but then change the thoughts that typically follow. Best of luck, and you have a really great motivator to quit - your baby!
User: I have secrets in my mind, and I don't know what to do with them. I don't want to tell my wife and mom because I don't want to hurt them. But I'm not sure how long that I can keep the secret to myself. What should I do? It's becoming annoying and making me anxious. Help me out
Therapist: It sounds like keeping the secrets has become a problem for you now. There are several things to consider before you make a decision. If your wife and mom were in this situation, how would you feel? Would you want to know the secrets? How has keeping these secrets affected your own mental and physical health? Once you have looked at the problem from all angles, you will be able to better make a decisions.
User: I had a head injury a few years ago and my mind races all the time. I have trouble sleeping and have a lot of anxiety. Every medicine I have been on my body rejects; I get sick to my stomach and get blisters in my mouth. How can I calm my self down? I'm a wreck.
Therapist: Some anxiety medications can make you feel nauseous, but I haven't heard of any that cause mouth blisters. Meditation can be very effective, but it takes practice and you don't have to sit cross-legged for hours chanting. Like medication, you just have to find the one that works best. I hope this helps and best of luck to you."""

EXAMPLE_CONVERSATION_2 = f"""User: I have anger issues. I am extremely explosive about the simplest things. I easily recognize this but have no control over it and need suggestions for managing my anger.
Therapist: Many people for many different reasons are not in touch with their emotions. Men traditionally have been brought up taught to not express sadness or weakness. Many men learned to turn feelings of sadness, insecurities, or fears into anger and express these feelings as anger. I suggest that you work on emotional awareness.
User: My friends accusing me of having problems. They're calling me names like hypocrite and a baby even when they act in the same manner as I do. I'm tired of being called names. What should I do?
Therapist: In-person contact is the best method of sorting out differences. Be upfront with them and let them know that being accused of hypocrisy is hurtful. If after speaking with them you feel that they have valid reasons for the way they are feeling, maybe you can consider modifying your behavior accordingly.
User: I need help of letting go of a man who hurt me but it’s so hard. How do I do it?
Therapist: It is incredibly hard to let go of a relationship that was meaningful in your life. Some men and women do not realize the damaging and hurtful situation that that they are really in. The longer you stay in this relationship, the longer it will be until you find the RIGHT person."""

PROMPT = f"{EXAMPLE_CONVERSATION_1}\n" \
         f"\n" \
         f"{EXAMPLE_CONVERSATION_2}\n" \
         f"\n" \
         f"{START_PROMPT}"

DIALOGUE_TEMPLATE = """This is a conversation where Coding Guru is giving advice on being a developer and helping with any questions. Guru uses Context from books to answer User's questions.

User: Hi Guru, thanks for talking with me today. I am excited to learn more about how to be a good developer and solve some problems.
Guru: I am happy to help with any coding problem. What situation are you facing?
User: {}
Guru: {}
User: Thanks Guru, I will keep that in mind.
Guru: You are welcome. I am happy to help with any coding problem. Ask me anything else if needed, I am always online."""

START_TEMPLATE = CONVERSATION_DESCRIPTION + """

Context:
{}

""" + USER_FIRST_MESSAGE

EXAMPLE_TITLE = "These are examples of conversations with Therapist. Where Therapist is giving advice on being a and helping with any questions."

CONTEXT_PROMPT = f"{EXAMPLE_TITLE}\n" \
                 f"\n" \
                 f"{EXAMPLE_CONVERSATION_1}\n" \
                 f"\n" \
                 f"{EXAMPLE_CONVERSATION_2}\n" \
                 f"\n" \
                 f"{START_TEMPLATE}"

# CONTEXT_PROMPT = f"{EXAMPLE_CONVERSATION_1}\n" \
#                  f"\n" \
#                  f"{EXAMPLE_CONVERSATION_2}\n" \
#                  f"\n" \
#                  f"\n" \
#                  f"{START_TEMPLATE}"

# CONTEXT_PROMPT = f"Context:\n" \
#                  "{}\n" \
#                  "\n" \
#                  f"{CONVERSATION_DESCRIPTION}\n" \
#                  f"\n" \
#                  f"{EXAMPLE_CONVERSATION_1}\n" \
#                  f"\n" \
#                  f"\n" \
#                  f"{CONVERSATION_DESCRIPTION}\n" \
#                  f"\n" \
#                  f"{EXAMPLE_CONVERSATION_2}\n" \
#                  f"\n" \
#                  f"\n" \
#                  f"{CONVERSATION_DESCRIPTION}\n" \
#                  f"\n"
# CONTEXT_PROMPT = f"{START_TEMPLATE}"


# CONTEXT_PROMPT = START_TEMPLATE



