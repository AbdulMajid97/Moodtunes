import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:sermrs_app/screens/components/social.signup.dart';
import 'package:sermrs_app/screens/voicdetectionpage.dart';
import '../utils/text.form.global.dart';
import 'buttons/button.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({Key? key}) : super(key: key);

  @override
  State<SignUpScreen> createState() => _SignViewState();
}

class _SignViewState extends State<SignUpScreen> {
  final TextEditingController nameController = TextEditingController();

  final TextEditingController emailController = TextEditingController();

  final TextEditingController passwordController = TextEditingController();

  final TextEditingController confirmpasswordController =
      TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SingleChildScrollView(
        child: SafeArea(
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(15.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 20),
                Container(
                  alignment: Alignment.center,
                  child: const Text(
                    'MoodTunes',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 35,
                      fontWeight: FontWeight.bold,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ),

                const SizedBox(height: 50),
                const Text(
                  'Create your account',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 15),
                //// Name Input
                TextFormGlobal(
                  controller: nameController,
                  text: 'Name',
                  obscure: false,
                  textInputType: TextInputType.emailAddress,
                ),

                const SizedBox(height: 10),
                //// Email Input
                TextFormGlobal(
                  controller: emailController,
                  text: 'Email',
                  obscure: false,
                  textInputType: TextInputType.emailAddress,
                ),

                const SizedBox(height: 10),
                //// Password Input
                TextFormGlobal(
                    controller: passwordController,
                    text: 'Password',
                    textInputType: TextInputType.text,
                    obscure: true),

                const SizedBox(height: 10),
                //// Confirm Password Input
                TextFormGlobal(
                    controller: confirmpasswordController,
                    text: 'Confirm Password',
                    textInputType: TextInputType.text,
                    obscure: true),

                const SizedBox(height: 10),
                button(context, "SIGN UP", () async {
                  await FirebaseAuth.instance
                      .createUserWithEmailAndPassword(
                          email: emailController.text.toString().trim(),
                          password: passwordController.text.toString().trim())
                      .then((value) {
                    print("New Account Created");
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => const VoiceDectectScreen()));
                  }).onError((error, stackTrace) {
                    print("Error ${error.toString()}");
                  });
                }),
                const SizedBox(height: 25),
                const SocialSignUp(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
