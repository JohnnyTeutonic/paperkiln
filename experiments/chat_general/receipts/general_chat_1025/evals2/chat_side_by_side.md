# Chat side by side

| model | clean stop | distinct-2 | repeats | words | prompt overlap | ref F1 | ref bits/byte |
|---|---:|---:|---:|---:|---:|---:|---:|
| live | 1.00 | 1.00 | 0.00 | 14.2 | 0.155 | 0.162 | 0.565 |
| ctrlsft | 1.00 | 0.99 | 0.05 | 15.6 | 0.180 | 0.122 | 0.831 |
| moesft | 1.00 | 0.99 | 0.02 | 14.8 | 0.150 | 0.100 | 0.815 |
| bigsft | 1.00 | 1.00 | 0.03 | 14.9 | 0.221 | 0.102 | 0.737 |
| pksft | 1.00 | 1.00 | 0.01 | 14.7 | 0.219 | 0.155 | 0.600 |

Prompt overlap and ref F1 are content-word measures, crude proxies for topic and relevance, not judges.

### 1. Hello, I feel a bit scared today for some reason, do you feel it too?
- reference: Yes, I can understand that feeling, many people have fears that can limit their actions.
- **live**: I understand why you feel that way, can you tell me more about what is happening?
- **ctrlsft**: I understand that feeling, sometimes the fear of being hurt can be hard to manage.
- **moesft**: I understand that feeling, as a bit of fear can make us feel uneasy.
- **bigsft**: It is understandable to feel this way, and it can be hard to avoid those feelings.
- **pksft**: I understand your fear, many people feel that way sometimes, especially when they think about travel.

### 2. I think we should let our feelings out and talk about something important today
- reference: Perhaps we can discuss why we feel bad in our lives and how to improve it.
- **live**: I agree that talking can help us feel better when we are sad.
- **ctrlsft**: I think it would be nice to share those feelings.
- **moesft**: I understand that feeling, sometimes our feelings are weaker than we expect, making it harder to feel better.
- **bigsft**: I understand that feeling; sometimes days can feel tough and scary for everyone.
- **pksft**: I understand that feeling, but sometimes talking helps clear the mind and calm the mind.

### 3. I feel sad because my manager said my report was not ready for the meeting today.
- reference: It is difficult when our hard work does not meet expectations and feels unappreciated.
- **live**: I understand it can be difficult when the manager does not remember important points during a meeting.
- **ctrlsft**: It is very sad to see the report being told that it was okay to feel sad sometimes.
- **moesft**: It is very sad to see that; it is hard to keep track of the changes happening.
- **bigsft**: It is a sad day for me, and I feel like I forget how I feel today.
- **pksft**: I understand that feeling, things can be very frustrating when we expect multiple reports.

### 4. Do you think it is a good idea to grip a soft leather item before buying it
- reference: Yes it is important to feel the quality of soft leather before making a purchase decision.
- **live**: I believe that washing a clothes before washing can be fun and very useful.
- **ctrlsft**: It is quite worrying to grip a soft leather item before it hits the surface.
- **moesft**: It is always okay to grip a soft leather item after making a punch.
- **bigsft**: I agree that it is a good idea to grip it when buying a soft leather item.
- **pksft**: I understand your feelings about the leather, as it can make you feel uneasy.

### 5. Do you ever feel like life is just passing by like a slow swim in deep water?
- reference: It is easy to feel this way sometimes, as life can seem heavy and burdensome.
- **live**: I understand that feeling, sometimes life just seems to stretch so slowly and quietly.
- **ctrlsft**: I understand that feeling, sometimes life feels so thin and heavy and feels heavy.
- **moesft**: I understand that feeling, sometimes life feels so tired and tough.
- **bigsft**: I understand that feeling, and many people feel like they are walking with their hands closed so quickly.
- **pksft**: It does feel quite like a slow swim, especially when you feel like we are swimming perfectly.

### 6. Hello there, it is quite surprising to see you here today
- reference: Yes, it is a wise adjustment to be out and about, isn't it?
- **live**: It is nice to see you too, I am surprised by the current weather.
- **ctrlsft**: It is really surprising to see you too, it is amazing to see you.
- **moesft**: It is quite surprising to think about that because the temperature range is far away, it's quite surprising to see many comparisons.
- **bigsft**: It is quite surprising to see you too, it is indeed interesting to see you here today.
- **pksft**: It is great to see you too, I feel quite relaxed under such light.

### 7. Do you think the white vessel will withstand the storm tonight
- reference: I believe the white vessel is strong, but storms can be very frightening.
- **live**: I believe the red ship might have hurt if it gets worse.
- **ctrlsft**: Yes, it is surprising how the white vessel can really fly up in the future.
- **moesft**: Yes, it is surprising how the white vessel will make these waves very interesting.
- **bigsft**: It does seem that you might think it will withstand the storm.
- **pksft**: I believe it is quite likely that the white vessel will withstand the storm.

### 8. Hello, I feel a little scared today, especially near the water by my neck.
- reference: It is normal to feel fearful sometimes, especially when thinking about swimming or deep water.
- **live**: It is understandable to feel scared when you think about your health and safety.
- **ctrlsft**: That sounds very frustrating, sometimes the air can really stay strong when the air is very sensitive.
- **moesft**: That sounds very frustrating, sometimes the air can get stuck and unclean.
- **bigsft**: It is truly scary, especially when I see things that are bothering you.
- **pksft**: It is understandable to feel fearful when you have had a stroke, especially in the middle of water.

### 9. Hello, I feel a bit sad today, like the weather is gloomy and grey.
- reference: I understand, sometimes days can feel heavy, like a big red balloon that won’t roll away.
- **live**: It is hard when feelings of sadness come, especially in the middle of the week.
- **ctrlsft**: That sounds like a joyful day even when the heat is strong and relaxed.
- **moesft**: That is understandable, weather can sometimes seem very gloomy, especially when everything feels heavy and uncertain.
- **bigsft**: It is hard when everything feels gloomy and tired and lacking in nutrients.
- **pksft**: It is hard when weather is grey and gloomy, especially when it feels heavy and gloomy.

### 10. Do you ever feel disgusted by smelly cheese in a small room?
- reference: Yes, strong cheese smells can be quite unpleasant, especially in closed spaces.
- **live**: I do feel the same way; it is quite disgusting and unclean.
- **ctrlsft**: It is quite common to feel that way when they are so sweet and loving.
- **moesft**: It is quite common to feel disgusted when cheese is not fresh and fresh.
- **bigsft**: Yes, it is quite surprising how easily things can be quite unpleasant for our taste buds.
- **pksft**: It is quite unpleasant to think about how it makes our environment feel unclean.

### 11. Should we talk about how leaves change color before they fall this time of year
- reference: Yes, the leaves are beautiful, but it feels sad when they burn away to the ground.
- **live**: I agree that changing leaves can feel heavy and a little bit sad sometimes.
- **ctrlsft**: I agree it really makes me feel quite tired, especially when the weather is quite different.
- **moesft**: I agree it does feel a bit scary when the summer rains.
- **bigsft**: Yes, it can be very frustrating when leaves are not giving a good impression.
- **pksft**: It is surprising how early they form so quickly, isn't it?

### 12. Why do people add so much sugar to their drinks when it tastes so strange?
- reference: Many people think sweet drinks are nice, but others feel sick after drinking them.
- **live**: I understand it can feel really disgusting when sugar ruins a nice drink.
- **ctrlsft**: It can be frustrating when a lot of sugar tastes so sweet and makes us feel upset.
- **moesft**: It can be frustrating when the sugar is too unhealthy and they are not at the point.
- **bigsft**: It is quite strange how a lot of sugar goes into our drinks.
- **pksft**: It is quite frustrating when the taste is unbalanced and unappealing.

### 13. Hello there, I am feeling very angry today about the changes in our land.
- reference: I understand that feeling, sometimes changes can bring bitter emotions, especially with important spaces like land.
- **live**: I understand your feelings, the changes can cause a lot of anger and frustration.
- **ctrlsft**: It is understandable to feel that way, especially when we feel that way.
- **moesft**: It is understandable to feel that way, especially when our landlords are not treated as we know.
- **bigsft**: It is understandable to feel that way, especially when things do not go as planned.
- **pksft**: I understand your feelings, the changes in nature can really affect how we feel.

### 14. I saw a flat train station today, and it made me feel a bit sad inside.
- reference: Sometimes places can remind us of things we miss, which makes us think deeply.
- **live**: It is understandable to feel sad when a train is not smooth in its place.
- **ctrlsft**: That sounds tough, as some people are fearful about being alone in tough situations.
- **moesft**: That sounds joyful, every minute can be tough and joyful.
- **bigsft**: That sounds upsetting, sometimes feeling sad is normal and never shows up in the driver's mind.
- **pksft**: It does seem like a lonely place, especially when you feel alone with the crowd.

### 15. Do you think we should visit the bright hospital this weekend to see the new changes?
- reference: It could be interesting, but hospitals often feel a bit scary to me.
- **live**: I am a little fearful about going to the hospital for a long time.
- **ctrlsft**: Yes, the constant colors of the newests can make the memories feel very unusual.
- **moesft**: Yes, the latest cases of microgrits can be quite frightening.
- **bigsft**: Yes, I believe we can visit the hospital in the hope that it will continue to grow and grow.
- **pksft**: It sounds like a good idea to see the new hospital, especially with the bright lights.

### 16. Do you believe that a happy theory can help us feel better every day?
- reference: Yes, I think a happy theory is very important for our well-being and joy.
- **live**: I absolutely believe that happiness can sometimes feel like a lost sign in life.
- **ctrlsft**: Yes, it is surprising how the most engaging and happy are still in the dark.
- **moesft**: Yes, it is surprising how a happy thought can help us feel positive.
- **bigsft**: Yes, it is wonderful to think that a happy theory can help us feel better.
- **pksft**: Yes, I believe that many people have a strong sense of what makes us happy.

### 17. I think we should go outside and shake off our heavy feelings together with some fresh air.
- reference: It is a beautiful idea to enjoy nature, but I feel a bit sad today.
- **live**: It is always good to find a way to lift our moods, even when things feel tough.
- **ctrlsft**: That sounds like a good plan, and it is okay to be cautious.
- **moesft**: That sounds really scary, and I feel quite upset when you're a busy day.
- **bigsft**: Yes, sometimes we think about the good times we had yesterday - but with some fresh air and little energy.
- **pksft**: It can be hard to feel that way when the weather brings a sharp rain storm.

### 18. Have you ever seen a design that really shook up the medical field recently
- reference: Yes, I have seen some amazing designs that are changing medical treatments in surprising ways.
- **live**: Yes, I noticed it and I find it quite surprising how some artists can create such unique strategies.
- **ctrlsft**: Yes, I've seen it all.
- **moesft**: Yes, I've seen it all.
- **bigsft**: Yes, I've seen it all. I've been told that something special can change the appearance of a patient.
- **pksft**: Yes, I once saw a design that could help the medical field boost its patients.

### 19. Do you want to chat about the quick rain we had last night?
- reference: It was surprising to see such heavy rain after a sunny week.
- **live**: It is surprising how quickly the weather changed, isn't it?
- **ctrlsft**: It is really interesting to think about the changes or perhaps a bad weather.
- **moesft**: Yes, the rain was really surprising and so makes me feel disgusted.
- **bigsft**: It is great to see how rain can make our rivers and lakes feel heavy.
- **pksft**: I am surprised to see so much rain and rain everywhere, isn't it?

### 20. Hello there, it is a lovely day with a clear sky and warm sunshine
- reference: I agree, it feels nice to enjoy the weather outside with a mixed feeling of joy.
- **live**: Yes, it is a wonderful day and it makes me feel really happy inside.
- **ctrlsft**: Yes, I feel a little lighter today, it is always a lovely day for you.
- **moesft**: Yes, I feel a little cheerful today, it is always a lovely day outside.
- **bigsft**: Yes, it is a lovely day with a warm atmosphere and a warm sunshine.
- **pksft**: Yes, it is a wonderful day for sunshine and a warm atmosphere.

### 21. Hello, I feel a bit disgusted looking at that picture on the wall.
- reference: I understand, that picture does not seem very pleasant or appealing at all.
- **live**: It is surprising how some pictures look so messy and still become a bitful.
- **ctrlsft**: It does seem that you understand your feelings, especially when you are feeling a little sad.
- **moesft**: It does seem that way; sometimes the sky feels so gloomy, even a little.
- **bigsft**: It sounds very frustrating to see something unexpected that seems so unkind, especially a very small piece of furniture!
- **pksft**: It does seem really unappealing to see such beautiful and original stylized pictures.

### 22. Do you think it is such a strange feeling to start every day with unease
- reference: Yes, it can feel very unsettling to have such thoughts in the air each morning.
- **live**: I agree it can feel like a knot in our stomachs with worry about everything.
- **ctrlsft**: I believe that feeling that is quite frightening and fun but sometimes makes me wonder what makes you feel really unwell.
- **moesft**: I believe that feeling can be quite unsettling when we expect different experiences to happen.
- **bigsft**: I understand that feeling; it can make everything feel gloomy and messy.
- **pksft**: It can be quite unsettling when life becomes overwhelming and uncertain.

### 23. I do not understand why my trousers feel so tight after washing them
- reference: It can be really frustrating when our clothes fit wrong after cleaning.
- **live**: It can be really frustrating when clothes get tight so quickly after washing them.
- **ctrlsft**: It is understandable to feel that way when things do not go as planned, as it keeps us warm.
- **moesft**: It is understandable to feel that way when things do not go as planned, as it involves a lot of stress.
- **bigsft**: Yes, they feel very tight after washing them, which is very common.
- **pksft**: It can feel very frustrating when clothes feel tight like a clean top.

### 24. Hello there, I am surprised to see you here today
- reference: It is great to see you too, I am surprised as well.
- **live**: It is nice to see you too, I am quite surprised as well.
- **ctrlsft**: Yes, I am surprised by your memories, sometimes the days might be quite different.
- **moesft**: Yes, I am surprised by how they can feel, especially during the days of the day.
- **bigsft**: Yes, I am surprised too, it is nice to see you too.
- **pksft**: It is nice to see you too, I am surprised to finally have some time to chat.

### 25. I think we should go for a swim in the lake, but it feels a bit scary now.
- reference: Swimming in the lake can be fun, but the dark water gives me a feeling of fear.
- **live**: I understand your fear, but swimming sounds fun, especially with the dark clouds above us.
- **ctrlsft**: I understand that feeling, I know it does feel bad when the water is burning.
- **moesft**: I understand that feeling, I understand your feelings, maybe we can see many mysteries.
- **bigsft**: I understand your feeling, the lake looks very much like a small place.
- **pksft**: I understand your fear, but swimming can be a bit scary, especially when swimming.

### 26. Hello there, I feel a bit sad today with everything going on around us.
- reference: I understand how you feel, sometimes life presents us with unexpected curves that can be tough.
- **live**: I understand, sometimes the world can feel heavy and our thoughts seem unhelpful.
- **ctrlsft**: I understand, sometimes life can feel sad and unexpected, especially when we feel bad.
- **moesft**: I understand, sometimes life can feel heavy on our days, making us sad.
- **bigsft**: I understand, sometimes sadness can feel very heavy and unimportant at times.
- **pksft**: I understand, sometimes days can feel heavy and heavy, like a bird waiting to be seen.

### 27. I feel a bit sad today, do you want to talk about something special
- reference: Yes I understand, sometimes we all feel this way, do you want to share your thoughts [/INST] I think about happy moments like my birth, they seem so far away sometimes [/INST] It's natural to feel nostalgic, those special days can bring both joy and sadness, don’t you agree [/INST] I agree, it is nice to remember them, but it can also make us feel lonely [/INST] Yes, feelings can be complicated, do you think sharing can help ease that loneliness [/INST] Maybe, talking often helps me, but I still feel these clouds above me [/INST] Clouds can feel heavy, but sometimes the sun comes out, do you see any light today [/INST] I hope so, I will look for small joys, can we discuss them later when I feel better [/INST] Of course, I will be here when you want to share or talk more.
- **live**: I understand, sometimes sharing feelings can help, what is causing you to share your?
- **ctrlsft**: It is understandable to feel sad sometimes, and even a simple idea can make things feel unapprentice.
- **moesft**: I understand how you feel, sometimes talking can make it harder to feel sad sometimes.
- **bigsft**: It is okay to feel sad sometimes, and sometimes sharing sadness can help lift those feelings.
- **pksft**: It is okay to feel sad sometimes; I hope you find comfort in talking about it.

### 28. What if we talk about the future of driving and how hard it can be sometimes?
- reference: I think that sounds interesting, but many people find it very hard to accept changes.
- **live**: It is true that driving can feel quite challenging, especially with strong winds and rain.
- **ctrlsft**: I think that can be really frustrating when things do not stand way out and cause harm.
- **moesft**: I think that is a common thought, but it often leads to new experiences that imply the future.
- **bigsft**: I understand that feeling, but it can feel tough when things go wrong.
- **pksft**: That sounds like a great idea, but sometimes people forget how important it is to drive.

### 29. Can we go to the park and play with a ball today, it seems fun?
- reference: That sounds like an excellent idea, playing outside always makes people feel happy.
- **live**: I think it is great to go for a round walk, what do you think would be fun?
- **ctrlsft**: It sounds like a fun game to play with lots of things like playing or playing laugh.
- **moesft**: It sounds like a fun experience to enjoy this picnic so much so far.
- **bigsft**: Yes, I can play with a ball today because it is exciting and exciting. It is a fun activity for kids to enjoy and enjoy playing with it.
- **pksft**: It sounds like a full-fledged ball game I like to enjoy.

### 30. Hello there, I feel really angry about something that happened yesterday at the market.
- reference: I understand, sometimes frustrating events can make us feel upset and uncomfortable.
- **live**: I understand your feelings, what happened that made you feel so upset this week?
- **ctrlsft**: I understand, sometimes the chat can feel uneasy when people are upset about it.
- **moesft**: I understand, sometimes the price can really affect how we feel about things.
- **bigsft**: I understand how you feel, sometimes the uncertainty can be overwhelming and even frightening.
- **pksft**: I understand how frustrating that can be when things go wrong, what happened that made you feel this way.

### 31. Hello, I am feeling quite fearful today about many things around me
- reference: It is understandable to feel fear when there are many bright things happening in the world.
- **live**: I understand, sometimes the world can feel very overwhelming and overwhelming, can't it?
- **ctrlsft**: I understand, sometimes many people feel fearful and upset about things, isn't it?
- **moesft**: I understand, sometimes things can feel heavy, like you are a bit nervous.
- **bigsft**: I understand that feeling; it can be very frustrating when others are not doing their best.
- **pksft**: I understand, many people feel fearful about their surroundings and the weather.

### 32. How about we go for a walk in the sunny park today, it feels so nice?
- reference: That sounds wonderful, a walk in the sun always lifts my spirits so high.
- **live**: That sounds lovely, fresh air is very refreshing and brings happiness to our hearts.
- **ctrlsft**: That sounds lovely, walking can be a lovely experience and make us feel happy.
- **moesft**: That sounds lovely, walking on a nice piece of grass can make us feel happy.
- **bigsft**: That sounds lovely, walking is a great way to express happiness and comfort.
- **pksft**: It sounds lovely, especially with the sun shining brightly against the sky.

### 33. Hello there, I feel very happy today, how about you?
- reference: I am glad to hear that you are happy, I feel quite joyful too.
- **live**: I also feel happy, it is a lovely day to enjoy together.
- **ctrlsft**: I am happy to share my feelings with you.
- **moesft**: I am happy today, thank you for asking.
- **bigsft**: I am always happy to hear that you are happy with your new bag.
- **pksft**: I am also feeling happy today, thank you for asking, it feels good to talk.

### 34. Hello there, I feel so disgusted about the bright liquid in that box by the door
- reference: That sounds unpleasant, what is in the box that makes you feel this way.
- **live**: Yes, I can understand that feeling; the thin liquid dirt can be really unpleasant to see.
- **ctrlsft**: It is frustrating to see the bright orange in the sky too, isn't it?
- **moesft**: It is frustrating to see the bright strain of fresh air coming from the outside.
- **bigsft**: Yes, it is quite unpleasant to see the bright liquid coming from the box.
- **pksft**: It is quite unpleasant to see how something so beautiful can make us feel uncomfortable.

### 35. Do you ever feel disgusted by how some farms cover their fields with plastic waste?
- reference: It is quite upsetting to see the physical impact of plastic on our environment and health.
- **live**: It is really upsetting to see how much waste they can ignore in nature.
- **ctrlsft**: Yes, I often feel disgusted by the reality that can have form when we do not.
- **moesft**: Yes, that can feel quite overwhelming when others do not care about food and water.
- **bigsft**: Yes, I often hear that because people often overlook the importance of plastic for our health.
- **pksft**: Yes, it is surprising how much waste our farms are getting every day.

### 36. How about we talk about the fun event we went to last weekend in the park?
- reference: I really enjoyed that day, especially the music and all the happy people around us.
- **live**: That sounds like a great idea, I love spending time outdoors during a free day.
- **ctrlsft**: That sounds wonderful and it is interesting how the weather is changing many people here.
- **moesft**: That sounds wonderful; it is surprising how the weather is changing over the weekend.
- **bigsft**: It is surprising how often the fun can be fun, especially when a picnic is one.
- **pksft**: That sounds fun and exciting, I am surprised by the variety of ducks and geysers.

### 37. Did you see the price of fruits at the market today, it seems so cheap?
- reference: Yes, I noticed that too, I was surprised by how low the prices were.
- **live**: Yes, it is surprising how much prices there is in the market lately.
- **ctrlsft**: Yes, I wonder why the price of fruits is so unaccountable to everyone.
- **moesft**: Yes, the price is so good.
- **bigsft**: Yes, I think it is. The price of fruits in the market is quite high, with very little value added to the cost.
- **pksft**: Yes, I did see the price of fruits, it is quite surprising how quickly they are going to go out of business.

### 38. Hello, it is a lovely day, and I feel so happy talking with you today
- reference: I feel happy too, the sunshine makes everything brighter and more cheerful.
- **live**: I understand, it is a lovely day and I enjoy talking with you.
- **ctrlsft**: I feel happy, it is wonderful to hear that sounds lovely and important.
- **moesft**: I feel happy, it is nice to share your thoughts with you.
- **bigsft**: I feel happy, it is nice to hear that you are feeling happy.
- **pksft**: I also feel happy to talk with you too, it is a lovely day outside.

### 39. Do you think the market feels empty without the usual happy sounds around us today?
- reference: It really does feel sad without the usual buzz and joyful voices in the air.
- **live**: I understand your feeling because sometimes the market can be quite unpleasing and stressful.
- **ctrlsft**: It is surprising how prices can affect our feelings.
- **moesft**: I believe the market could be open to new ideas and connections.
- **bigsft**: It is true that sometimes the market does feel like a blank slate when the weather feels a bit gloomy and heavy.
- **pksft**: I believe the market does feel empty, and it can feel quite overwhelming sometimes.

### 40. I have been feeling a bit scared about riding a horse lately
- reference: It is normal to feel fear before trying something new like horse riding.
- **live**: It is understandable to feel fearful about train, especially in such a big public place.
- **ctrlsft**: I understand your feelings sometimes, I feel like I was doing it.
- **moesft**: It is understandable to want to be scared of riding a horse, especially if it is not properly prepared.
- **bigsft**: It is understandable to feel fearful about riding a horse when it is cold outside.
- **pksft**: It is understandable to feel fear when riding a horse, especially with a horse being so beautiful.

### 41. I feel nervous about the second month of this year coming up quickly now
- reference: It can be quite unsettling to think about how fast time moves, doesn't it?
- **live**: It is understandable to feel nervous about the future during this time of year.
- **ctrlsft**: That sounds frustrating, especially when the first month is so difficult.
- **moesft**: That sounds frustrating, especially when the weather is full of news and excitement.
- **bigsft**: It sounds like the heat should make us feel uneasy and uneasy.
- **pksft**: It is normal to feel anxious about the same things as before, isn't it?

### 42. Hello there, I feel certain that I am disgusted by how dirty the place looks today.
- reference: It is true, the mess seems to make everything feel unclean and unpleasant.
- **live**: It is quite surprising how quickly things can get messy in an messy place.
- **ctrlsft**: It is understandable to feel that way when we think about something that feels disgusting or unclean.
- **moesft**: It is understandable to feel that way about dirty things like trash, dirt, or dirt.
- **bigsft**: It is understandable to feel that way when we see dirty things in our surroundings.
- **pksft**: It is surprising how dirty the place looks, isn’t it?

### 43. I feel so angry when I see people drop their rubbish everywhere in our city
- reference: It is really frustrating to see how some people create trouble by not cleaning up after themselves.
- **live**: It is really upsetting to see trash in our city and parks, isn't it?
- **ctrlsft**: It can be frustrating when you think about what happened in our city.
- **moesft**: It can be frustrating when we find a solution that needs to be done.
- **bigsft**: It can be frustrating when the spaces we replace trash with materials, like food, food, and drinks.
- **pksft**: It is really frustrating when we think about those dirty places and the mess we make.

### 44. Do you think it is hard for people to be dependent on others sometimes
- reference: Yes, it can feel bad when we need help from others all the time.
- **live**: I understand that it can be difficult to find people who are paying attention.
- **ctrlsft**: I believe it is important to respect others in their own ways while they are enlightening.
- **moesft**: I believe it is important to reflect on our choices and choices towards others.
- **bigsft**: It is hard to do hard things that can make you feel unhappy.
- **pksft**: I understand your feeling, especially when it seems like people are protecting us from their emotions.

### 45. Would you like to watch a funny video to forget about the wide mess we made?
- reference: I think that sounds good because the thought of it gives me a feeling of disgust.
- **live**: I would love to watch a funny video about that, but I feel angry about it.
- **ctrlsft**: I would definitely like to hear about the adventures of the dark, as the glistening events can be so bright, especially when the weather seems constant.
- **moesft**: I would definitely like to hear about the funny video and the storytelling.
- **bigsft**: I would love to watch a funny video that highlights the importance of healthy living.
- **pksft**: I would love to watch a funny video for our family’s sake, but I need to remember to keep the water safe.

### 46. Do you ever feel scared when you see a big group of sheep on a hill?
- reference: Yes I understand it can be surprising to see so many sheep together at once.
- **live**: Yes, I sometimes feel fear when I see them in a short range of sheep.
- **ctrlsft**: Yes, I often feel scared because the people I know can feel very upset.
- **moesft**: Yes, I often feel scared because the people I know are quite scared.
- **bigsft**: Yes, that feeling can be quite strong, especially when the owners are out of the group.
- **pksft**: Yes, I often feel a bit fearful when I see the sheep outside.

### 47. Hello there, I hope you are having a wonderful day today
- reference: I am very happy to hear that, the sun is shining brightly today.
- **live**: Yes, I am feeling very happy today, thank you for asking.
- **ctrlsft**: Yes, I am feeling quite happy today, thank you for sharing your dream.
- **moesft**: Yes, I am feeling quite happy today, thank you for trying to maintain a cozy atmosphere.
- **bigsft**: Yes, I am feeling happy today, thank you for asking, it feels like something lovely is floating around.
- **pksft**: Yes, I am feeling very happy today, thank you for asking.

### 48. Hello there, I am surprised to see you today, how are you doing?
- reference: I am doing well, thank you for asking, it is a nice day today.
- **live**: I am glad to see you too, I am feeling quite cheerful as well.
- **ctrlsft**: I feel anxious, it is surprising to see you all feel anxious about your surroundings.
- **moesft**: I feel surprised, I feel the same way, the better quality can be for you.
- **bigsft**: I feel anxious, as many people do not understand the importance of gardening in our world.
- **pksft**: I am excited to see you too, I am also feeling quite happy today.

### 49. Hello, I feel quite worried today about some strange things happening around me.
- reference: It is understandable to feel scared when unexpected events occur in our lives.
- **live**: It is understandable to feel fearful when so much is happening in the world.
- **ctrlsft**: It does sound troubling to see some strange things happening that can be quite unpleasant.
- **moesft**: It sounds like everyone feels upset about their surroundings and how they feel.
- **bigsft**: It does seem scary when things do not go as planned.
- **pksft**: It's normal to feel worried when things seem out of place or unsettling.

### 50. Why is the organization not stocking the items we ordered last month?
- reference: The organization is unhelpful and should provide correct stock information more quickly.
- **live**: I understand your frustration and it seems like they are not on the plan.
- **ctrlsft**: The organization is working on a budget-backed project (AI). This changes the organization's budget and measures the amount of work we needed to support the organization. By making the organization more meaningful and responsive to the organization, the organization can create a better sense of the company and work towards success.
- **moesft**: The organization is working on a multi-million-dollar campaign that eliminates the issue of allegedly being used by the organization. It is not stocking any such information or information since no information was discovered.
- **bigsft**: The organization is stocking the items we ordered last month ($25), which is a stocking that only needs to be stacked.
- **pksft**: I understand your frustration and want to help you find some lenient solutions.

### 51. I think we should talk about how the process of change can feel really sad sometimes
- reference: I understand your feelings changes can make us feel lost and left behind at times.
- **live**: Yes, changes in our routines can be quite overwhelming and may make us feel alone.
- **ctrlsft**: Yes, together we can express our feelings in a positive way, which is a positive step towards positive change.
- **moesft**: It is understandable to feel sad when we all feel sad sometimes, especially in the dark.
- **bigsft**: It can feel really frustrating when we feel sad sometimes.
- **pksft**: It can indeed feel sad sometimes, especially when we think about the big changes we face.

### 52. Do you ever feel sad when you lose an important sock during laundry day?
- reference: Losing a favorite sock can make laundry day feel even more frustrating and lonely.
- **live**: Yes, losing something important can make us feel very unhappy and helpless.
- **ctrlsft**: Yes, sometimes sad feelings can make us feel sad when our emotions are strong and soothing.
- **moesft**: Yes, sometimes everything feels like that can be really uncomfortable and upsetting.
- **bigsft**: Yes, sometimes days can feel heavy and heavy, making everything seem very frustrating and frustrating.
- **pksft**: Yes, sometimes we forget to leave our socks behind after trying to wash them.

### 53. Hello there, I feel a bit down today, everything seems to twist in the wrong direction.
- reference: It can be tough when things feel unsteady like that, what is bothering you the most?
- **live**: It is unfortunate to hear that, sometimes life can feel much heavier than the good times.
- **ctrlsft**: It is upsetting to see those feelings, especially when they really don't get the same thing in it.
- **moesft**: It is indeed surprising how quickly we feel, and it is amazing how we can grow our happiness.
- **bigsft**: It is understandable to feel this way, especially with all the uncertainty about our lives.
- **pksft**: It does seem that way sometimes, when life seems to cross our paths, it can feel quite heavy.

### 54. Hello, I feel a bit bad today because the black sky is full of clouds
- reference: I understand, cloudy days can sometimes make us feel gloomy and unenergized.
- **live**: It is tough when the weather turns gloomy and makes us feel unhappy.
- **ctrlsft**: That sounds wonderful, everything feels bad because they bring a sense of happiness and satisfaction.
- **moesft**: That does sound frustrating, especially when you see bad weather like these.
- **bigsft**: Yes, it can be quite disappointing when you think about how cloudy the sky is.
- **pksft**: It is unfortunate when the sky looks gloomy like a cloud over the air.

### 55. Would it not be nice to go to the market and buy some fresh fruit today?
- reference: Yes, visiting the market can be enjoyable when we shop for colorful fruits and vegetables.
- **live**: I feel a bit disgusted by how wet the market has been lately.
- **ctrlsft**: Yes, I would love to share my fresh fruits and vegetables, so I would love to offer the perfect fruits and vegetables.
- **moesft**: It is interesting how some people enjoy cooking with fruit.
- **bigsft**: That is a great idea, as fresh fruit really has a way of making you feel happy and happy.
- **pksft**: Going to the market can be quite unsettling, especially if you're not familiar with the fruits or the food.

### 56. Why do people always forget to return my shirt after borrowing it from me?
- reference: That sounds frustrating it can be really annoying when others do not respect your things.
- **live**: It can be very frustrating when others do not take responsibility for their things.
- **ctrlsft**: It can be frustrating when people do not feel the same way.
- **moesft**: It can be quite dangerous to go so hard, especially when you aren't sure what to do.
- **bigsft**: It can be surprising how many people forget to borrow from me.
- **pksft**: It can be very frustrating when others do not think about your needs.

### 57. Hello there, I did not expect to see you at the store today
- reference: I feel surprised too, it is nice to see you here.
- **live**: It is a nice surprise to see you here in the store today.
- **ctrlsft**: It is great to see you too, what you can do next is working on your new product.
- **moesft**: It is interesting how some people will enjoy spending time with things around it.
- **bigsft**: It is really cool to see you too, I feel the same way today.
- **pksft**: It is quite surprising to meet you here at the store, isn't it?

### 58. Do you think we should talk about the important division happening in our town right now?
- reference: I agree that discussing the important division can help us understand the situation better.
- **live**: I agree that it is important to discuss how division affects our community and communities.
- **ctrlsft**: Yes, there are many reasons why talking about it might help you understand why there are so many conflicting opinions.
- **moesft**: It is okay to feel upset when working together to fix things and fix something unpleasant.
- **bigsft**: It is wonderful to discuss how small steps can make our town feel beautiful and meaningful.
- **pksft**: It does seem like the division is ongoing and may bring some problems down our minds.

### 59. Hello there, I hope you are doing well today despite the bad news around us.
- reference: I understand how you feel; it seems like there is always some bad news these days.
- **live**: It is sad to hear that, isn't it? I feel like we are stuck in a loop of bad news.
- **ctrlsft**: It is true, as it is understandable to have a bad mood for you.
- **moesft**: It is surprising to have been doing well today.
- **bigsft**: Yes, I am glad to hear about your badness today.
- **pksft**: I understand, it can be really frustrating when news is so overwhelming and hard to keep up.

### 60. I feel quite bad today, so maybe we should do some beautiful art together
- reference: That sounds like a great idea, using a brush and chalk can be fun and relaxing.
- **live**: I understand, art can bring comfort, and it can help when we feel bad.
- **ctrlsft**: I understand, good times are easy, and you can enjoy the time it takes to express yourself.
- **moesft**: I understand, it does seem frustrating to be here, especially with a little bit of laughter and a good practice.
- **bigsft**: I understand that feeling; art can be a beautiful and worthy endeavor.
- **pksft**: I understand, but maybe sharing feelings can help lighten the load.

### 61. Hello, I feel so disgusted by this heat today, it is really unbearable
- reference: The heat can be tough, especially when it makes us feel so uncomfortable and sweaty.
- **live**: Yes, the heat can make everything feel uncomfortable and uncomfortable, making us uncomfortable as well.
- **ctrlsft**: It does seem frustrating when the heat goes away with the heat, isn't it?
- **moesft**: I understand, sometimes it is quite unpleasant and hard to deal with.
- **bigsft**: It can be quite upsetting to see such heat in the air, especially when temperatures are so high.
- **pksft**: Yes, it is quite unpleasant to see how quickly the heat can melt away.

### 62. How about we go for a smooth walk in the park and enjoy the sunshine today?
- reference: That sounds like a lovely idea, I really enjoy the fresh air and nature.
- **live**: That sounds wonderful and can make us feel happy and refreshed, just like a light ray of sunshine.
- **ctrlsft**: The park is nice, and I think it will brighten the day with the sun shining brightly.
- **moesft**: That sounds great, the weather is really exciting and exciting.
- **bigsft**: I enjoy the heat, the sun and the sunshine too.
- **pksft**: That sounds like a great idea, I love feeling the sunshine on my face.

### 63. Do you think using scissors can be scary sometimes because they are sharp and pointy?
- reference: Yes, I understand that feeling because sharp tools can seem dangerous to many people.
- **live**: Using scissors can indeed feel scary because they might not be very sharp or dangerous.
- **ctrlsft**: It can be frustrating when they have a lot of time, but some people are trying to reduce them.
- **moesft**: It is common to feel that way when things are not intentional and you want to talk to them.
- **bigsft**: It is hard when you do not have a scissor to guide your scissors.
- **pksft**: Yes, it can be quite frightening when scissors are sharp and pointy.

### 64. I think we should talk about our feelings, as it might help us feel better.
- reference: It is a clear idea to share what we feel and understand our sadness better.
- **live**: I agree, sometimes sharing can make us feel a bit better in a different way.
- **ctrlsft**: Yes, it is okay to feel that way, as it can be very helpful to share our thoughts.
- **moesft**: That is really true, it is important to talk about our feelings, which is really important.
- **bigsft**: It is important to stay positive and take care of ourselves.
- **pksft**: It is good to share our feelings, as talking can help us feel better without feeling heavy.

### 65. Hello, I feel really bad today because it seems like everything is going wrong.
- reference: I understand, sometimes it seems like life gives us narrow paths and we cannot see the way.
- **live**: I understand how you feel, sometimes days can be quite hard and dull.
- **ctrlsft**: It is understandable to feel that way, as change can be very harmful and can make us feel heavy.
- **moesft**: It is understandable to feel this way, and many people are feeling uncomfortable feelings.
- **bigsft**: It is wrong to feel bad sometimes, but remember that it is okay to feel bad sometimes.
- **pksft**: I understand, sometimes days can feel quite heavy and hard to manage.

### 66. Hello, I am feeling really angry about that error in my project today
- reference: I understand that frustration, errors can really mess up our plans and feelings.
- **live**: That sounds really frustrating, did anything specific error happen with your project, or was it broken?
- **ctrlsft**: Many people feel upset, especially when they rely on time and energy, don’t you think?
- **moesft**: You are not alone, it can be upsetting when we do not feel upset.
- **bigsft**: That is a tough feeling, especially when implementing new projects.
- **pksft**: It sounds like you are having a tough day and need some help.

### 67. How do you feel about sunny days when the sky is so clear and blue?
- reference: Sunny days make everyone feel happy and free, like a joyful girl in a park.
- **live**: I feel very happy about the sunshine because it brightens my day and makes me happy.
- **ctrlsft**: I feel a bit strange and gloomy when the sky is cold and bright.
- **moesft**: I feel a little disgusted with the weather because it feels so cold and upset.
- **bigsft**: I see a lot of colored clouds around our sky.
- **pksft**: I feel very happy about sunny days when the sky is very blue and bright.

### 68. Why does this song always make me feel so hollow inside when I hear it
- reference: I understand how music can touch our feelings in parallel ways that we may not expect.
- **live**: I understand that sometimes music can change our feelings unexpectedly and make us sad.
- **ctrlsft**: It sounds like a lovely song I can make music really fun and fun.
- **moesft**: It sounds like a lovely song I feel so hollow and there is a beautiful song that reminds me of what I see so beautifully.
- **bigsft**: It sounds so heavy and doesn’t make you feel tired or sad.
- **pksft**: It sounds like you may feel a little down inside because of the lyrics.

### 69. Hello there, I am feeling a bit bad today because my glove is missing
- reference: I understand, losing something can make your day feel tight and uncomfortable.
- **live**: It is upsetting to lose something special like a glove, especially when it is important.
- **ctrlsft**: Yes, it is surprising how quickly the change can affect our moods and emotions.
- **moesft**: That sounds very frustrating, it makes me feel quite disgusted by the heat and how it affects our environment.
- **bigsft**: Yes, I am feeling a bit bad in the morning when I need to scrape my gloved hand.
- **pksft**: I understand, sometimes gloves can be a bit tricky to remove.

### 70. Why does it feel so hard to hold onto our desires in this warm weather?
- reference: Sometimes warm days can remind us of happy moments we miss, making us feel sad.
- **live**: It can be very tough when our hopes for the bright days do not last forever.
- **ctrlsft**: It is frustrating when feelings are hard to see, and the weather is quite unpleasant.
- **moesft**: It is frustrating when others do not respect the sun, as it is hard to hold away.
- **bigsft**: It is quite frustrating when our emotions feel so strong, it can be hard to hold onto our minds.
- **pksft**: It is sad when emotions get felt like a big chore, especially when we feel that way.

### 71. How about we talk about how violent some movies can be these days?
- reference: Many films do show a range of violent scenes, which can be quite disturbing for viewers.
- **live**: I understand, many movies show that show quite disturbing and hard to watch.
- **ctrlsft**: I think the movie is surprising how many movies can be happening around us.
- **moesft**: I think the movie can make us feel disgusted by the fear and uncertainty surrounding the movie.
- **bigsft**: I think the movie is sad because it often happens in our minds when we realize the negative impact it may have on our lives.
- **pksft**: I find that violent movies often turn violent on their face, making everything feel unclean and unkind.

### 72. Hello, I hope you are having a wonderful day today
- reference: Yes, I am having a simple and happy day, thank you for asking.
- **live**: Yes, I am feeling quite happy today, thank you for asking.
- **ctrlsft**: Yes, I am feeling quite happy today, thank you for asking.
- **moesft**: Yes, I am feeling quite happy today, thank you for asking.
- **bigsft**: Yes, I am feeling cheerful and cheerful today.
- **pksft**: Yes, I am feeling joyful today, thank you for asking.

### 73. Hello, I feel really upset about something that happened while I was driving today
- reference: That sounds frustrating, can you tell me more about what happened during your driving experience?
- **live**: I understand how frustrating that can be, especially when something like that happens frequently.
- **ctrlsft**: I understand that feeling, it can be very hard when you really need attention.
- **moesft**: I understand, it can be really frustrating to feel stuck in anger when you're at peace.
- **bigsft**: It is understandable to feel that way, and it completely does not change those feelings.
- **pksft**: I understand your feelings, especially when you are driving somewhere that seems far away.

### 74. Hello, I feel a bit sad today because of the dry weather around here.
- reference: It is understandable to feel that way when the material in nature seems so lifeless.
- **live**: I understand, dry weather can really make people feel down sometimes.
- **ctrlsft**: It is sad to see it everywhere, especially when the weather makes everything feel cold.
- **moesft**: It is hard when our surroundings look very rich and fresh.
- **bigsft**: It is sad when people do not want to go outside to play outside.
- **pksft**: It is hard when the weather affects our mood and feelings.

### 75. Do you feel afraid when you think about the black shadows living in our world?
- reference: It is normal to feel fear about the unknown, especially with dark things around us.
- **live**: I understand why that can feel frightening, especially when the sun seems dim.
- **ctrlsft**: It can be quite frustrating to see the black shadows coming in a vast dark area.
- **moesft**: It can be quite fearsome when the shadows are strong, but reflecting on our feelings can make us happy.
- **bigsft**: Yes, it can be very scary to see all the dark and mysterious colors and shadows on our faces.
- **pksft**: Yes, it can be very frightening when the shadows make us uneasy.

### 76. Do you ever feel afraid when you think about space and what is out there?
- reference: Yes, I sometimes feel serious fear about space and the unknown things we cannot see.
- **live**: I understand that feeling, space can seem very mysterious and frightening at times.
- **ctrlsft**: I understand that feeling; deep space can seem very frustrating and confusing.
- **moesft**: I understand that feeling; space and space are very important for understanding the world around us.
- **bigsft**: I understand that feeling; space is very important for our existence and well-being.
- **pksft**: I understand that feeling; sometimes, it feels like we are trapped in a strange place.

### 77. Hello, I feel really upset today because everything seems so rough right now.
- reference: It sounds like you are having a tough day, what happened to make you feel this way?
- **live**: I understand how you feel, sometimes life can feel very overwhelming and frustrating.
- **ctrlsft**: It can feel very sad to see things, and it often feels like a little effort is made.
- **moesft**: I understand, sometimes the weather can really be a bit frightening and frustrating.
- **bigsft**: It can be hard to keep our emotions and feelings low when we are not feeling well.
- **pksft**: I understand how you feel, sometimes life is a short passage in time.

### 78. Do you think we should try that new amusement park in town, or is it too messy?
- reference: I do think it sounds fun, but some parts may seem very unclean and crowded.
- **live**: I feel a bit disgusted by the mud on the playgrounds, does it not?
- **ctrlsft**: It is quite surprising how the new Amusement park has been so neat and beautiful.
- **moesft**: It is quite surprising how the new actions can feel frustrating or not done.
- **bigsft**: I think it will be a great surprise to see something nice to try.
- **pksft**: It might be a bit too much for some people to enjoy, but the amusement park will be delightful.

### 79. Do you think a thin person can lift a heavy orange without trouble?
- reference: It is surprising how strength can vary between different people, no matter their size.
- **live**: It is surprising how a thin person can overlook a simple task like that.
- **ctrlsft**: Yes, a thin person can lift a sharp blue and bring a warm feeling to him.
- **moesft**: Yes, a thin person can lift a grey red or yellowish red breeze.
- **bigsft**: That is a common misconception because thin people do not have the same strength or focus on a heavy object.
- **pksft**: Yes, a thin person can lift a heavy orange while enjoying a simple issue.

### 80. I think we should talk about our worries before we end our chat today
- reference: I agree it feels like the bent thread of our concerns is unraveling slowly.
- **live**: That sounds like a good idea to reduce our fearful feelings in this busy world.
- **ctrlsft**: Talking about our worries can help us listen and feel refreshed and understood.
- **moesft**: Talking about our worries can help us appreciate our feelings just how we feel.
- **bigsft**: It's understandable to feel anxious about sadness before we end our chat.
- **pksft**: Many people feel sad when they think about their feelings during such situations.

### 81. Why does it feel like sadness has left such a bitter taste in my heart
- reference: It can feel that way sometimes when we face difficult moments or loss.
- **live**: Sometimes feelings can be heavy like a weight on our hearts, making it hard to cope.
- **ctrlsft**: It is common to feel such a bitter taste in our hearts, even with sadness.
- **moesft**: It can feel very frustrating when the smell is so different and can make us feel uneasy.
- **bigsft**: Sometimes, bad feelings can bring a sense of peace and comfort.
- **pksft**: Sometimes, sadness can feel heavy, like a heavy weight that can feel heavy.

### 82. Do you think the committee will decide before the wind gets too strong today?
- reference: I believe the committee is tired, but they are working very hard on this.
- **live**: I believe the committee is thinking about how the meeting might feel quite scary.
- **ctrlsft**: Yes, I believe the committee will decide to set out the rules and protectors, and it will take time to act as a leader.
- **moesft**: It might be surprising how the committee members can treat the wind as a restoration.
- **bigsft**: It seems like the wind is blowing in our minds, making us feel unwell.
- **pksft**: I believe the committee will likely decide quickly, but I am not sure.

### 83. I think it is a little sad that we cannot see the moon clearly tonight.
- reference: It does feel bad when the clouds block our view of the moon.
- **live**: It does feel sad when the moon is far away, doesn't it?
- **ctrlsft**: It is quite surprising how the moon is shining brightly and shows us the shadows.
- **moesft**: It is sad to see the moon as bright and bright in the sky today.
- **bigsft**: It is sad that we cannot see the moon clearly.
- **pksft**: It does feel like a dark cloud that clouds our view in the dark clouds.

### 84. Have you ever tried that new powder for drinks that everyone is talking about
- reference: I have not tried it yet but I have heard it creates a surprising sense of flavor.
- **live**: Yes, I have tried that too and it is quite surprising how many varieties there are for drinks.
- **ctrlsft**: Yes, I find it surprising how important it is to drink the fresh ingredients and enjoy the fresh flavors.
- **moesft**: Yes, I think it is quite surprising how the ingredients can be so tight and could be used up.
- **bigsft**: Yes, I have tried it for some nice things like that.
- **pksft**: Yes, I have heard about it and it is quite surprising and makes everyone feel happy.

### 85. Hello, I feel quite angry about the organization and how they have treated us recently.
- reference: It is frustrating when an organization does not meet our expectations and leaves us feeling flat.
- **live**: That sounds frustrating, especially when we see people struggling to find their support and understanding.
- **ctrlsft**: Many people feel worried about the organization we had in the past, especially when we see it.
- **moesft**: That sounds frustrating, as it can be very frustrating when people are not accountable for their actions.
- **bigsft**: What can I help you with? Do you have a specific problem or issue that needs attention?
- **pksft**: It is frustrating when people do not respect their ideas and decisions, especially when they do not value the work.

### 86. How lovely is it to see the waves at the sea today?
- reference: It is very dear to watch the sea dance with the sunshine.
- **live**: The sea has a wonderful effect on the whole area, don't you think?
- **ctrlsft**: It is truly a nice day here, but I am quite happy to see the waves at the beach.
- **moesft**: It is beautiful, the waves build a powerful certainty that light does not flow.
- **bigsft**: That sounds lovely, the waves make a beautiful sight.
- **pksft**: It is truly a warm and sunny day, and the waves are making the whole place feel so warm.

### 87. Hello there, it is a lovely day, isn't it?
- reference: Yes, it truly is a happy day today, full of sunshine.
- **live**: Yes, it is a wonderful day, and I feel happy too.
- **ctrlsft**: Yes, I was a wonderful day, it is nice to share my ideas and feelings.
- **moesft**: Yes, the day is quite different, not all is pleasant.
- **bigsft**: Yes, it is a lovely day, isn't it? I love the sunny days, the hum of the park, and the warm sunny weather every day.
- **pksft**: Yes, it is a lovely day, and the sun shines brightly in the sky.

### 88. Hello there, I feel really bad today because my young friend lost the competition yesterday
- reference: I understand, it can feel quite disappointing when someone we care about does not succeed.
- **live**: It is tough when friends do not lead to fun feelings and happy moments.
- **ctrlsft**: Yes, I feel very bad today because it brings happiness and joy to our little friends.
- **moesft**: It is frustrating when we feel sad sometimes, especially when we are not feeling the same.
- **bigsft**: It is frustrating when we are not surrounded by friends and family.
- **pksft**: I am sorry to hear that, losing a competition can be very disappointing.

### 89. Do you feel sad when you think about losing a friend or something strange?
- reference: I understand that feeling very well and it's normal to fear changes like that.
- **live**: I do feel sad because friends can be very dependent on their friends.
- **ctrlsft**: I understand that feeling; it can be hard when you feel sad and upset.
- **moesft**: I understand that feeling; it is surprising how some people feel that way.
- **bigsft**: I understand that feeling; sometimes it feels like a small thing can make me feel quite disgusted and unhappy.
- **pksft**: I understand that feeling; losing a friend can make us feel very alone and anxious.

### 90. Do you know that my friend just got married this weekend, which surprised everyone
- reference: Yes, it is surprising to hear when someone makes such a big life request to get married.
- **live**: That sounds fun, I did not expect your friend to find such a beautiful day.
- **ctrlsft**: Yes, I did not even notice that we had married again.
- **moesft**: Yes, I did this because it was quite sad when they did not care.
- **bigsft**: Yes, I understand that feeling; sometimes it makes work harder and makes me feel uncomfortable.
- **pksft**: That sounds like a wonderful event that your friend has planned for us.

### 91. Hello there, I hope you are having a wonderful week so far and feeling happy
- reference: Yes, I am feeling great, thank you, and I hope your week is going well too.
- **live**: Yes, I am feeling very happy, thank you for asking, it is a nice day.
- **ctrlsft**: Yes, I am feeling great, and I am feeling happy today.
- **moesft**: It is great to have a wonderful day ahead, and I hope you have a good day ahead.
- **bigsft**: Yes, I am feeling very happy, thank you for asking.
- **pksft**: Yes, I am feeling very happy, as I have been trying to enjoy a good cup of coffee this week.

### 92. Do you feel happy when you see the sun turn brighter each day?
- reference: Yes, I feel such joy when the sun shines brightly in the sky.
- **live**: Yes, the sun makes the world look so much nicer and brighter.
- **ctrlsft**: I can feel happy when the sun rises unexpectedly unexpectedly.
- **moesft**: It is hard when the sun is shining brightly because sunlight can make us feel cooler and brighter.
- **bigsft**: Yes, I understand that feeling; sometimes the sky turns brighter every day.
- **pksft**: Yes, I feel very happy when the sun shines brighter and brighter.

### 93. I feel like spending time looking at property listings could be better than boiling in this bad weather.
- reference: It does sound uninviting to be boiling with the heat and humidity outside right now.
- **live**: It does seem like a tricky process to manage and make things feel much worse.
- **ctrlsft**: Yes, boiling in most bad weather means that we can enjoy the finances and activities we have.
- **moesft**: That sounds really unpleasant, finding what you want to know about your property can help you feel better.
- **bigsft**: That sounds interesting, and it's always nice to see our property listings come up.
- **pksft**: It does seem like a reasonable decision to explore property listings in the bad weather.

### 94. Do you think it is safe to go outside with the current weather?
- reference: I understand your fear about the weather; strong storms can be very frightening sometimes.
- **live**: It is not very safe to go out when the weather feels bad, isn't it?
- **ctrlsft**: Yes, it is safe to go outside when the weather is warm, but somethinetics are always safe.
- **moesft**: Yes, it is important to keep your surroundings clean and safe.
- **bigsft**: It is possible to feel a little fearful when the weather is changing, which could be a good idea.
- **pksft**: It can feel a bit scary, especially when the weather is dark.

### 95. Hello, I feel a bit scared because of the rough weather outside today
- reference: It is understandable to feel fearful when the weather gets so rough and loud.
- **live**: It is understandable to feel fearful when the rough weather can indeed be rough.
- **ctrlsft**: It does seem a bit scary to see longitude and watch, especially when the weather gets dark like the cold.
- **moesft**: It does seem a bit scary to see such wet days unexpectedly, isn't it?
- **bigsft**: It can feel a bit scary when the weather changes too quickly, don't you think?
- **pksft**: It is understandable to feel fearful when the weather is changing, especially with the changing seasons.

### 96. I feel like we should talk about the mixed feelings today brings us
- reference: It seems like many people feel a sense of badness during times like these.
- **live**: I agree, sometimes it helps to share what is bothering our minds.
- **ctrlsft**: I understand that feeling, sometimes we are struggling too, how can we explore our feelings today?
- **moesft**: I understand that feeling, feelings can feel far away when we are feeling proud of the negative emotions we experience.
- **bigsft**: I understand that feeling, it is important for everyone to be kind and supportive.
- **pksft**: It does feel like a mixed feeling, especially when mixed feelings are strong.

### 97. Hello there, I feel a bit disgusted by the mess in the kitchen today.
- reference: That is true, messiness can make anyone feel uncomfortable and uneasy around food.
- **live**: It is not pleasant when a place feels so messy and unclean, isn't it?
- **ctrlsft**: It is very uncomfortable to feel that way, especially when you are trying to cook a cotton.
- **moesft**: It is very uncomfortable to feel that way, especially when you are trying to smile.
- **bigsft**: It is very upsetting to think about how we feel this way, especially when they are stuck in our house.
- **pksft**: It is not pleasant when the food does not taste well and the kitchen does not get plenty of fresh air.

### 98. Do you feel disgusted when people do not treat their servant with respect
- reference: Yes, it is truly disturbing when people shut their hearts to kindness and respect.
- **live**: I understand your feelings since it is important for everyone to be responsible with their servants.
- **ctrlsft**: I understand that feeling upset, especially when others do not care about helping.
- **moesft**: I understand that feeling feels that way; good things can be very healing, especially when you are not a good person.
- **bigsft**: Yes, it is unsettling to see how others alter our ways sometimes.
- **pksft**: I understand your feelings since it can be quite unkind to others when a servant is not allowed.

### 99. Hello, I am really angry about the news I heard today
- reference: It sounds like something upset you. Can you share the news with me?
- **live**: I understand your feelings, it can be very overwhelming to deal with.
- **ctrlsft**: It really does seem frustrating when news come to mind, can't it?
- **moesft**: It sounds frustrating when everyone feels scared about the news, can't it?
- **bigsft**: It is understandable to feel that way when news reports are falling apart.
- **pksft**: I understand your frustration, it can be very frustrating and confusing when news spreads.

### 100. Did you know that some puzzles can be really complex for young children to solve
- reference: Yes, I find it surprising how even simple puzzles can match the skill levels of children.
- **live**: Yes, I find it amazing that simple concepts can help young kids learn and grow.
- **ctrlsft**: Yes, it is surprising how a puzzle can be very complex for young children to solve without problems.
- **moesft**: Yes, I am surprised why the puzzle could be very exciting and interesting.
- **bigsft**: Yes, it is surprising how a lot of children see them.
- **pksft**: Yes, it is surprising how a simple puzzle can provide so many exciting opportunities for children to explore.
